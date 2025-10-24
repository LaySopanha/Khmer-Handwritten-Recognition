import os
from typing import List, Tuple


def detect_lines_paddle(image_path: str, use_angle_cls: bool = False) -> List[List[Tuple[float, float]]]:
    """Detect text regions using PaddleOCR DB detector.

    Returns a list of polygons (each polygon is a list of (x, y)).
    Requires `paddleocr` to be installed.
    """
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        raise RuntimeError(
            "PaddleOCR not installed. Install with `pip install paddleocr`"
        ) from e

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    ocr = PaddleOCR(use_angle_cls=use_angle_cls, use_gpu=False, show_log=False)
    result = ocr.ocr(image_path, det=True, rec=False)
    boxes: List[List[Tuple[float, float]]] = []
    for line in result:
        for det in line:
            poly = det[0]
            boxes.append([(float(x), float(y)) for x, y in poly])
    return boxes


def detect_stub(image_path: str) -> List[List[Tuple[float, float]]]:
    return []

