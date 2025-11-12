#!/usr/bin/env python3
"""
Batch-run PaddleOCR detections and export Label Studio predictions.

Example (serve images over HTTP like LayoutLMv3):
    python annotation_tools/auto_annotate_with_paddle.py \\
        --images-dir data/raw \\
        --output notebooks/output/ls_preannotations.json \\
        --ls-mode http \\
        --ls-root data/raw \\
        --ls-prefix http://localhost:8081/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib.parse import quote
from uuid import uuid4

from paddleocr import TextDetection
from PIL import Image

DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Label Studio predictions using PaddleOCR detections."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing images to annotate.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the Label Studio JSON predictions.",
    )
    parser.add_argument(
        "--model-name",
        default="PP-OCRv5_server_det",
        help="PaddleOCR detection model name (default: %(default)s).",
    )
    parser.add_argument(
        "--ls-mode",
        choices=("local-files", "http", "storage"),
        default=os.environ.get("LABEL_STUDIO_MODE", "local-files"),
        help="How to build image URLs for Label Studio.",
    )
    parser.add_argument(
        "--ls-root",
        default=os.environ.get("LABEL_STUDIO_ROOT"),
        help="Root directory Label Studio can access (defaults to cwd).",
    )
    parser.add_argument(
        "--ls-prefix",
        default=os.environ.get("LABEL_STUDIO_PREFIX", "/data/local-files/?d="),
        help="URL prefix for local-files/http modes.",
    )
    parser.add_argument(
        "--from-name",
        default="lines",
        help="Label Studio control name (PolygonLabels/RectangleLabels).",
    )
    parser.add_argument(
        "--to-name",
        default="image",
        help="Label Studio object tag name (Image).",
    )
    parser.add_argument(
        "--label",
        default="text",
        help="Label value to attach to each polygon.",
    )
    parser.add_argument(
        "--model-version",
        default="paddle-ppocrv5",
        help="Model version string stored with predictions.",
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=list(DEFAULT_EXTS),
        help="File extensions to include. Default: %(default)s",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images (useful for smoke tests).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when gathering images.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def find_images(root: Path, exts: Sequence[str], recursive: bool) -> List[Path]:
    exts_lower = tuple(e.lower() for e in exts)
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_lower]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts_lower]
    files.sort()
    return files


def build_image_entry(
    path: Path,
    *,
    mode: str,
    ls_root: Path,
    prefix: str,
) -> str:
    try:
        rel = path.resolve().relative_to(ls_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Image {path} is outside of Label Studio root {ls_root}. "
            "Adjust --ls-root accordingly."
        ) from exc
    if mode == "http":
        prefix = prefix.rstrip("/") + "/"
        return f"{prefix}{quote(rel.as_posix())}"
    if mode == "storage":
        return rel.as_posix()
    # local-files handler
    return f"{prefix}{quote(rel.as_posix())}"


def extract_polygons(det_output: Sequence, img_size: Sequence[int]) -> List[dict]:
    img_w, img_h = img_size
    raw = det_output[0]
    res_dict = raw["res"] if isinstance(raw, dict) and "res" in raw else getattr(raw, "res", raw)
    if res_dict is None:
        return []
    dt_polys = res_dict.get("dt_polys") if isinstance(res_dict, dict) else getattr(res_dict, "dt_polys", None)
    dt_scores = res_dict.get("dt_scores") if isinstance(res_dict, dict) else getattr(res_dict, "dt_scores", None)
    if dt_polys is None or dt_scores is None:
        return []
    polys = dt_polys.tolist() if hasattr(dt_polys, "tolist") else dt_polys
    scores = dt_scores.tolist() if hasattr(dt_scores, "tolist") else dt_scores
    polygons = []
    for poly, score in zip(polys, scores):
        normalized = [
            {"x": round(x / img_w * 100, 4), "y": round(y / img_h * 100, 4)}
            for x, y in poly
        ]
        polygons.append({"points": normalized, "score": float(score)})
    return polygons


def create_prediction_result(polygons: List[dict], args: argparse.Namespace) -> List[dict]:
    results = []
    for poly in polygons:
        points = [[pt["x"], pt["y"]] for pt in poly["points"]]
        results.append(
            {
                "id": str(uuid4()),
                "from_name": args.from_name,
                "to_name": args.to_name,
                "type": "polygonlabels",
                "score": poly["score"],
                "value": {
                    "points": points,
                    "polygonlabels": [args.label],
                },
            }
        )
    return results


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    images_dir = Path(args.images_dir).resolve()
    if not images_dir.is_dir():
        raise NotADirectoryError(f"{images_dir} is not a directory")

    ls_root = Path(args.ls_root or os.getcwd()).resolve()
    if not ls_root.exists():
        raise FileNotFoundError(f"Label Studio root {ls_root} does not exist")
    if args.ls_mode != "storage" and not args.ls_prefix:
        raise ValueError("ls-prefix must be set for local-files/http modes")

    images = find_images(images_dir, args.exts, args.recursive)
    if args.limit:
        images = images[: args.limit]
    if not images:
        logging.warning("No images found in %s with extensions %s", images_dir, args.exts)
        return

    logging.info("Found %d images. Loading PaddleOCR model %s ...", len(images), args.model_name)
    detector = TextDetection(model_name=args.model_name)

    tasks = []
    for idx, img_path in enumerate(images, 1):
        logging.info("[%d/%d] Processing %s", idx, len(images), img_path.name)
        with Image.open(img_path) as img:
            img_size = img.size
        det_output = detector.predict(str(img_path), batch_size=1)
        polygons = extract_polygons(det_output, img_size)
        if not polygons:
            logging.debug("No polygons detected for %s; skipping", img_path)
            continue
        image_entry = build_image_entry(
            img_path, mode=args.ls_mode, ls_root=ls_root, prefix=args.ls_prefix
        )
        task = {
            "data": {"image": image_entry},
            "predictions": [
                {
                    "model_version": args.model_version,
                    "result": create_prediction_result(polygons, args),
                    "score": sum(p["score"] for p in polygons) / len(polygons),
                }
            ],
        }
        tasks.append(task)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    logging.info("Wrote %d tasks to %s", len(tasks), output_path)


if __name__ == "__main__":
    main()
