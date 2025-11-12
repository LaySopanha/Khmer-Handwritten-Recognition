# Project Context

## Objective
- Build a line-level Khmer handwritten OCR pipeline that uses PaddleOCR detectors for layout and Microsoft TrOCR for recognition.
- Automate page annotation so detection proposals can be reviewed quickly, then fine-tune TrOCR on curated Khmer snippets (lines or short sentences).

## Current Components
- Detection (`detector/`):
  - `infer_detect.py` wraps PaddleOCR DB to produce polygons per page.
  - `merge_boxes.py` groups word boxes into reading-order line boxes.
  - `crop_lines.py` exports padded crops plus a manifest for traceability.
  - `batch_detect_crop.py` ties everything together for folder-level batching.
- Recognition (`ocr/`):
  - `data.py`, `metrics.py`, `train_trocr.py`, `infer_trocr.py` give a full fine-tuning and inference loop around TrOCR.
  - `generation_config.json` stores default generation params for inference.
- Tokenizer (`tokenizer/build_vocab.py`) builds a Khmer char vocab JSON for a custom processor if needed.
- Annotation utilities (`annotation_tools/`) support Label Studio conversions.
- `KHMER_TROCR_MASTER_PLAN.md` is the detailed execution roadmap.

## Annotation & Data Status
- Source data: scanned notebook pages with Khmer handwriting.
- Current plan: run PaddleOCR detection to auto-propose line boxes, crop them, then review/fix text in PPOCRLabel or Label Studio.
- Crops + transcripts should be stored as JSONL rows `{"image_path": "...", "text": "..."}` to plug into `ocr/train_trocr.py`.

## Open Items
- Validate PaddleOCR DB on representative pages (double-check slanting/overlapping diacritics).
- Decide whether to keep per-line or switch to smaller snippets depending on recognition accuracy and annotation throughput.
- Finalize tokenizer choice (default byte-level vs. custom Khmer char vocab) before the main training run.
- Stand up an annotation workflow (PPOCRLabel or Label Studio) that can import the auto-detected boxes and export JSONL for training.

## Upcoming Tasks
- Dry-run `detector/batch_detect_crop.py` on a small batch; inspect crops and merged lines.
- Configure PPOCRLabel project folders so it reads crops/boxes and writes corrected transcripts.
- Produce initial train/valid splits (few thousand lines) and launch a baseline TrOCR fine-tune.
- Track decisions/results here to keep Codex and collaborators aligned.

## Label Studio Serving Notes
- Use `serve_local_files.py` (repo root) to expose any image directory with permissive CORS: `python serve_local_files.py --directory data/raw --port 8080`.
- For bulk preannotations run `annotation_tools/auto_annotate_with_paddle.py`:
  ```
  python annotation_tools/auto_annotate_with_paddle.py \
      --images-dir data/raw \
      --output notebooks/output/ls_preannotations.json \
      --ls-mode http \
      --ls-root data/raw \
      --ls-prefix http://localhost:8080/
  ```
  Adjust ports/paths for local-files or storage workflows.
- In notebooks/scripts exporting predictions set:
  - `LABEL_STUDIO_MODE=http`
  - `LABEL_STUDIO_ROOT` to the served directory
  - `LABEL_STUDIO_PREFIX=http://localhost:8080/`
- This mirrors the LayoutLMv3 workflow and avoids `/data/local-files/?d=` issues when switching Conda environments or machines.
