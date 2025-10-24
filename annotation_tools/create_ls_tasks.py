import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(description="Create Label Studio tasks JSON from a folder of images")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="", help="Optional prefix (e.g., /data/local-files/?d=")
    args = ap.parse_args()

    imgs = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        imgs.extend(glob.glob(os.path.join(args.images_dir, ext)))
    imgs = sorted(imgs)

    tasks = []
    for p in imgs:
        rel = os.path.relpath(p, args.images_dir)
        url = f"{args.prefix}{rel}" if args.prefix else rel
        tasks.append({"data": {"image": url}})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(tasks)} tasks to {args.out}")


if __name__ == "__main__":
    main()

