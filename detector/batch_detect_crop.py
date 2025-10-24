import argparse
import glob
import os
from typing import List, Tuple

from .infer_detect import detect_lines_paddle
from .crop_lines import crop_and_save
from .merge_boxes import merge_polys_to_line_boxes


def run_folder(
    pages_dir: str,
    out_dir: str,
    pattern: str = "*.{jpg,jpeg,png}",
    merge_to_lines: bool = True,
    pad: int = 2,
):
    os.makedirs(out_dir, exist_ok=True)
    # glob with multiple extensions
    paths: List[str] = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
        paths.extend(glob.glob(os.path.join(pages_dir, ext)))
    manifest_lines: List[str] = []
    for page_path in sorted(paths):
        polys = detect_lines_paddle(page_path)
        if merge_to_lines:
            from .merge_boxes import poly_to_bbox

            line_boxes = merge_polys_to_line_boxes(polys)
            # convert back to polys as simple 4-pt rects for cropper?
            polys_use = [
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for (x0, y0, x1, y1) in line_boxes
            ]
        else:
            polys_use = polys
        base = os.path.splitext(os.path.basename(page_path))[0]
        page_out_dir = os.path.join(out_dir, base)
        crop_paths = crop_and_save(page_path, polys_use, page_out_dir, pad=pad)
        for cp in crop_paths:
            manifest_lines.append(f"{cp}\t{page_path}")
    manifest = os.path.join(out_dir, "manifest.tsv")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("crop_path\tpage_path\n")
        f.write("\n".join(manifest_lines))
    print(f"Saved crops under: {out_dir}\nManifest: {manifest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages_dir", required=True, help="Folder of page images")
    ap.add_argument("--out_dir", required=True, help="Output folder for crops")
    ap.add_argument("--no_merge", action="store_true", help="Do not merge word boxes into lines")
    ap.add_argument("--pad", type=int, default=2)
    args = ap.parse_args()
    run_folder(args.pages_dir, args.out_dir, merge_to_lines=not args.no_merge, pad=args.pad)


if __name__ == "__main__":
    main()

