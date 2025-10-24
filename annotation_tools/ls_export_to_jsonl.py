import argparse
import json
import os


def extract_text_from_result(result):
    # Expecting a textarea result
    for r in result:
        if r.get("type") in {"textarea", "text"}:
            val = r.get("value", {}).get("text")
            if isinstance(val, list) and val:
                return val[0]
            if isinstance(val, str):
                return val
    return ""


def main():
    ap = argparse.ArgumentParser(description="Convert Label Studio JSON export to training JSONL")
    ap.add_argument("--ls_export", required=True, help="Label Studio JSON export file")
    ap.add_argument("--images_dir", required=True, help="Base dir of images in training set")
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    with open(args.ls_export, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for task in data:
        # LS stores the image path in task['data']['image']
        img_val = task.get("data", {}).get("image", "")
        # If using Local Storage, image may be like "/data/local-files/?d=subdir/img.png"
        # Try to extract relative path after '?d='
        if "?d=" in img_val:
            rel = img_val.split("?d=", 1)[1]
        else:
            rel = img_val
        # Normalize path
        rel = rel.replace("\\", "/")

        # Get chosen annotation (first completion or last, depending on your LS flow)
        anns = task.get("annotations") or task.get("completions") or []
        if not anns:
            continue
        ann = anns[-1]
        text = extract_text_from_result(ann.get("result", []))
        if not text:
            continue
        # Verify image exists
        abs_img = os.path.join(args.images_dir, rel)
        if not os.path.exists(abs_img):
            # Fallback: maybe LS stored absolute path already
            abs_img = rel
            if not os.path.exists(abs_img):
                # Skip missing
                continue
            # Recompute rel to be relative to images_dir
            try:
                rel = os.path.relpath(abs_img, args.images_dir)
            except Exception:
                pass
        rows.append({"image_path": rel, "text": text})

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()

