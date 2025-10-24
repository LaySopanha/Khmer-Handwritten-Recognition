from typing import List, Tuple


Box = Tuple[int, int, int, int]  # x0, y0, x1, y1


def poly_to_bbox(poly: List[Tuple[float, float]]) -> Box:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _iou_y(a: Box, b: Box) -> float:
    y0 = max(a[1], b[1])
    y1 = min(a[3], b[3])
    inter = max(0, y1 - y0)
    h_union = max(a[3] - a[1], 1) + max(b[3] - b[1], 1) - inter
    return inter / max(1, h_union)


def _merge_overlapping_in_row(boxes: List[Box], gap_ratio: float = 0.5) -> List[Box]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged: List[Box] = []
    cur = list(boxes[0])
    for b in boxes[1:]:
        cur_h = max(cur[3] - cur[1], 1)
        gap = b[0] - cur[2]
        # if horizontally close relative to height, merge
        if gap <= gap_ratio * cur_h:
            cur[2] = max(cur[2], b[2])
            cur[1] = min(cur[1], b[1])
            cur[3] = max(cur[3], b[3])
        else:
            merged.append(tuple(cur))
            cur = list(b)
    merged.append(tuple(cur))
    return merged


def merge_polys_to_line_boxes(polys: List[List[Tuple[float, float]]], row_iou_thresh: float = 0.1, gap_ratio: float = 0.5) -> List[Box]:
    """Group word-level polys into line-level boxes by y-overlap and horizontal proximity.

    Args:
        polys: list of polygons from detector
        row_iou_thresh: y-overlap based grouping threshold
        gap_ratio: max gap between adjacent boxes as a fraction of row height
    Returns:
        List of merged line-level bounding boxes.
    """
    boxes = [poly_to_bbox(p) for p in polys]
    # sort by y
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2.0)
    rows: List[List[Box]] = []
    for b in boxes:
        placed = False
        for row in rows:
            # compare with first box in row for speed
            if _iou_y(row[0], b) >= row_iou_thresh:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    merged_rows: List[Box] = []
    for row in rows:
        row = sorted(row, key=lambda r: r[0])
        merged_rows.extend(_merge_overlapping_in_row(row, gap_ratio=gap_ratio))
    # final sort top-to-bottom, left-to-right
    merged_rows = sorted(merged_rows, key=lambda b: (b[1], b[0]))
    return merged_rows

