import os
from typing import List, Tuple
from PIL import Image


def _poly_to_bbox(poly: List[Tuple[float, float]]):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def crop_and_save(image_path: str, boxes: List[List[Tuple[float, float]]], out_dir: str, pad: int = 2) -> List[str]:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    crop_paths: List[str] = []
    for i, poly in enumerate(boxes):
        x0, y0, x1, y1 = _poly_to_bbox(poly)
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x1 + pad)
        y1 = min(h, y1 + pad)
        crop = img.crop((x0, y0, x1, y1))
        out_path = os.path.join(out_dir, f"crop_{i:05d}.png")
        crop.save(out_path)
        crop_paths.append(out_path)
    return crop_paths

