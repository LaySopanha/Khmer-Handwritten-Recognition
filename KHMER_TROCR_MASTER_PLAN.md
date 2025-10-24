# Khmer Handwritten OCR with TrOCR — Master Plan

This document is a practical, step‑by‑step plan to build a Khmer handwritten OCR system using TrOCR for recognition and a separate detector for page layouts. It assumes a GPU machine with Python 3.9+.

---

## 0) Goals and Scope

- Build a production‑ready line‑level Khmer handwritten OCR pipeline:
  - Detection: find text regions in page images (lines preferred; words optional).
  - Recognition: fine‑tune TrOCR to transcribe cropped Khmer text.
  - Metrics: Character Error Rate (CER) as primary; optional WER if you define word boundaries.
- Keep the plan compatible with `pip install transformers` (no library source modifications required).

---

## 1) Environment Setup

- Install core packages (GPU):
  - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`  (adjust CUDA version as needed)
  - `pip install transformers datasets accelerate evaluate pillow regex numpy scipy` 
  - Detector option A (recommended to start): `pip install paddleocr`  (installs Paddle + detector; CPU works, GPU optional)
  - Detector option B (alternative): [MMOCR] — if you prefer, follow their install docs.
  - Utilities: `pip install albumentations jiwer rich opencv-python`
- Verify GPU: run `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.

Project workspace suggestion:

```
project_root/
  data/
    train/ images/  (line crops)
    valid/ images/
    test/  images/
    train.jsonl
    valid.jsonl
    test.jsonl
  detector/
    infer_detect.py        (page → boxes)
    crop_lines.py          (boxes → crops + reading order)
  tokenizer/
    build_vocab.py         (Option B only)
    khmer_vocab.json       (Option B only)
  ocr/
    train_trocr.py         (training entrypoint)
    data.py                (dataset + collator)
    metrics.py             (CER/grapheme CER)
    infer_trocr.py         (recognize crops/pages)
    generation_config.json (optional defaults)
  configs/
    train_base.yaml        (hyperparams, paths)
  KHMER_TROCR_MASTER_PLAN.md
```

---

## 2) Decide Recognition Granularity

- Prefer line‑level recognition for handwriting.
  - More context, less segmentation ambiguity than per‑word.
  - Works well with TrOCR’s sequence decoder.
- If your documents are extremely structured (e.g., printed forms with boxes), word‑level can be OK.

---

## 3) Data Collection and Normalization

- Gather real Khmer handwritten data with transcripts. If starting from pages, plan to detect + crop lines.
- Synthetic bootstrapping (optional): generate rendered Khmer lines using multiple fonts plus degradations.
- Normalize transcripts consistently:
  - Unicode normalization NFC.
  - Normalize whitespace (trim, collapse repeats if desired).
  - Decide whether to keep punctuation/numerals; be consistent across train/valid/test.
- Dataset format (JSONL):

  Example row in `data/train.jsonl`:

  ```json
  {"image_path": "images/000123.png", "text": "ខ្ញុំសរសេរភាសាខ្មែរ"}
  ```

- Splits:
  - Ensure deterministic split; balance lengths and rare characters across train/valid.
  - Place image files under the split’s `images/` folder.

---

## 4) Page Detection → Line Crops (if training from pages)

- Start with PaddleOCR DB detector for good generalization:
  - Input: page image
  - Output: polygons/rotated boxes for text regions
- Pipeline:
  1. Run detector to get boxes.
  2. Deskew each crop to horizontal; add small padding.
  3. Sort crops by reading order (top-to-bottom, left-to-right; consider row grouping).
  4. Save crops to `data/{split}/images/` and write matching JSONL with ground truths.
- Quality checks:
  - Visualize overlays (boxes on page) for random samples.
  - Spot check deskew and padding don’t cut diacritics.

Tip: If your pages are simple single-column notes, try projection-profile line segmentation (faster; fewer deps). Switch to DB detector only if needed.

---

## 5) Tokenizer Strategy

You do not need to modify Transformers for Khmer. Choose one of:

- Option A (quick baseline): Use TrOCR’s default byte-level BPE tokenizer.
  - Pros: zero setup, supports all Unicode via bytes.
  - Cons: less efficient for Khmer; slightly worse accuracy sometimes.
- Option B (recommended): Character-level Khmer tokenizer.
  - Build a vocabulary covering Khmer letters, diacritics, numerals, common punctuation, plus special tokens `[PAD] [BOS] [EOS] [UNK]`.
  - Wrap with `PreTrainedTokenizerFast` and use in a `TrOCRProcessor`.

Minimal steps for Option B:

1) Generate vocab from transcripts (unique code points you expect):

```
python tokenizer/build_vocab.py \
  --train_jsonl data/train.jsonl \
  --valid_jsonl data/valid.jsonl \
  --out tokenizer/khmer_vocab.json
```

`khmer_vocab.json` example keys (illustrative):

```json
{
  "bos_token": "<bos>",
  "eos_token": "<eos>",
  "pad_token": "<pad>",
  "unk_token": "<unk>",
  "vocab": ["<pad>", "<bos>", "<eos>", "<unk>", "ក", "ខ", "គ", "ឃ", "ង", "ា", "ិ", "ី", "។", "space"]
}
```

2) In code, load vocab, create `PreTrainedTokenizerFast`, and pass into `TrOCRProcessor`.

---

## 6) Processor and Image Policy

- Image policy for OCR crops:
  - Fixed height, keep aspect ratio; pad/clamp width.
  - Start with height = 384; max width cap ~ 1536 (tune as needed).
  - RGB, normalize to ViT mean/std.
- Processor setup (Option A or B):
  - `TrOCRProcessor(image_processor=ViTImageProcessor(...), tokenizer=...)`.

---

## 7) Model Initialization

- Start from a pretrained TrOCR checkpoint:
  - `microsoft/trocr-base-handwritten` (good default)
  - or `microsoft/trocr-base-stage1` (encoder pretraining focus)
- If using custom tokenizer (Option B):
  - Set `model.config.vocab_size = len(tokenizer)`
  - `model.resize_token_embeddings(len(tokenizer))`
  - Set `decoder_start_token_id`, `pad_token_id` from tokenizer
- Training trick: freeze encoder for 1–3 epochs, then unfreeze for joint fine-tuning.

---

## 8) Data Loader, Collator, and Augmentations

- Dataset class: reads JSONL rows, loads images, applies normalization + augmentations, encodes labels.
- Collator:
  - Pads pixel values to max width in batch.
  - Pads labels; set `-100` on padded label positions to ignore in loss.
- Augmentations (mild; keep legibility):
  - Small rotation (±2–3°), perspective jitter, brightness/contrast shift, light Gaussian noise, slight blur.

---

## 9) Training Configuration

- Core hyperparameters (starting point):
  - Optimizer: AdamW, weight decay 0.01
  - LRs: decoder 1e-4; encoder 1e-5 (lower LR on encoder)
  - Schedule: cosine with 5–10% warmup
  - Batch size: depends on VRAM (e.g., 8–16 at H=384); use gradient accumulation if needed
  - Mixed precision: `fp16` or `bf16`
  - Label smoothing: 0.1
  - Max target length: 128–256
  - Eval: every epoch; select best by validation CER
  - Early stopping: patience 3–5 evals on CER
- Generation params for eval: `num_beams=5`, `early_stopping=True`, reasonable `max_new_tokens`.

---

## 10) Metrics: CER (and Optional WER)

- CER at code point level is primary for Khmer.
- For better Khmer handling, compute grapheme-aware CER using `regex` and `\X` to segment grapheme clusters.
- Normalize both GT and predictions identically (NFC, whitespace rules).

---

## 11) Validation and Debugging

- Track:
  - Overall CER and CER by length buckets.
  - Loss curves for train vs. valid.
  - Qualitative spot checks: visualize a small grid of crops with GT vs. predicted text.
- Common failure modes:
  - Diacritic stacking confusion: consider more data/augs.
  - Long lines truncated: raise `max_new_tokens` or image width cap.
  - Overfitting: increase weight decay/label smoothing, reduce LR, add data.

---

## 12) Inference Pipeline (Pages and Crops)

- For crops:
  - Batch through `processor` → `model.generate(...)` → decode.
- For pages:
  1. Detect boxes (DB detector) → polygons/angles
  2. Deskew + crop; sort by reading order
  3. Recognize each crop; aggregate into page-level text (lines joined by `\n`)
- Confidence scoring:
  - Use sequence log-prob normalized by length.
  - Flag low-confidence lines for review.

---

## 13) Packaging and Deployment

- Save artifacts:
  - `pytorch_model.bin`, `config.json`, tokenizer files, `preprocessor_config.json`, `generation_config.json`.
- Version control checkpoint directories; optionally push to private HF Hub.
- Runtime:
  - PyTorch inference server or script.
  - For latency, consider ONNX export for the encoder; full ONNX possible but more involved.

---

## 14) Data Growth and Active Learning

- Loop:
  - Collect low-confidence lines from production.
  - Re-label; add to training set.
  - Periodically fine-tune from latest checkpoint.
- Monitor coverage of rare Khmer glyph combinations; curate additional samples when needed.

---

## 15) Risks and Mitigations

- Insufficient line segmentation quality → Improve detector or switch to line segmentation tailored to your documents.
- Sparse coverage of rare diacritics → Targeted data collection; synthetic augmentation with rare combos.
- Over-augmentation degrading readability → Dial back; keep augs mild.
- VRAM constraints → Lower height (e.g., 320), gradient accumulation, gradient checkpointing.

---

## 16) Suggested Timeline (Milestones)

- Week 1: Environment ready; small dataset assembled (≥3k lines); baseline training with Option A tokenizer; first CER result.
- Week 2: Build Option B tokenizer; retrain; CER improvement; integrate detector for page inference; end-to-end demo.
- Week 3: Hyperparameter tuning; refine augmentations; add grapheme-aware CER; finalize v1 checkpoint.
- Week 4+: Active learning loop; production hardening; deploy.

---

## 17) Execution Checklist (Copy/Paste)

- Environment
  - [ ] Install `torch`, `transformers`, `datasets`, `accelerate`, `evaluate`, `pillow`, `regex`.
  - [ ] Install detector (`paddleocr` or MMOCR).
  - [ ] Verify CUDA.
- Data
  - [ ] Collect/clean transcripts; normalize (NFC).
  - [ ] Create `data/{train,valid,test}/images` and JSONLs.
  - [ ] If from pages: detect → crop → save → verify overlays.
- Tokenizer/Processor
  - [ ] Option A (baseline) or Option B (char-level vocab) selected.
  - [ ] Build/load tokenizer; set special tokens.
  - [ ] Create `TrOCRProcessor` with `ViTImageProcessor` params (height, padding, mean/std).
- Model/Training
  - [ ] Load `microsoft/trocr-base-handwritten`.
  - [ ] If custom tokenizer: `resize_token_embeddings`, set `decoder_start_token_id`, `pad_token_id`.
  - [ ] Freeze encoder for 1–3 epochs, then unfreeze.
  - [ ] Configure Trainer: LRs, warmup, batch size, fp16, label smoothing, eval schedule.
  - [ ] Track CER; save best by CER.
- Evaluation/Inference
  - [ ] Implement code-point and grapheme-aware CER.
  - [ ] Build page-level pipeline: detect → crop → recognize → assemble.
  - [ ] Confidence scoring; sample review.
- Packaging
  - [ ] Save model + processor + generation config.
  - [ ] Optional push to HF Hub (private).
- Growth
  - [ ] Low-confidence relabeling loop.
  - [ ] Coverage tracking for rare glyphs.

---

## 18) Implementation Notes and Snippets

Below are concise code patterns to accelerate implementation. Adapt paths to your project layout.

- Load baseline TrOCR (Option A), set IDs, quick eval:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ckpt = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(ckpt)
model = VisionEncoderDecoderModel.from_pretrained(ckpt)

model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
model.config.vocab_size = model.config.vocab_size  # unchanged for Option A
```

- Swap to custom tokenizer (Option B):

```python
from transformers import ViTImageProcessor, TrOCRProcessor, PreTrainedTokenizerFast
import json

with open("tokenizer/khmer_vocab.json", "r", encoding="utf-8") as f:
    spec = json.load(f)

tokenizer = PreTrainedTokenizerFast(
    bos_token=spec["bos_token"], eos_token=spec["eos_token"],
    pad_token=spec["pad_token"], unk_token=spec["unk_token"],
    tokenizer_object=None,  # build from vocab mapping below
)
# Map tokens to ids in order
vocab_list = spec["vocab"]
id_to_token = {i: tok for i, tok in enumerate(vocab_list)}
token_to_id = {tok: i for i, tok in enumerate(vocab_list)}
# Assign to fast tokenizer internals
tokenizer._tokenizer = None  # Fast path if you construct via tokenizers lib; otherwise use simple mapping helpers
# Alternatively: use tokenizers library to build a WordLevel/Unigram tokenizer and wrap it.

img_proc = ViTImageProcessor(size={"height": 384}, do_resize=True, do_normalize=True)
processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tokenizer)

model.config.vocab_size = len(vocab_list)
model.resize_token_embeddings(len(vocab_list))
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
```

- CER metric (grapheme-aware):

```python
import regex as re

def graphemes(s):
    return re.findall(r"\X", s)

def cer(pred, ref):
    import numpy as np
    p = graphemes(pred)
    r = graphemes(ref)
    # Levenshtein distance
    dp = [[0]*(len(r)+1) for _ in range(len(p)+1)]
    for i in range(len(p)+1): dp[i][0] = i
    for j in range(len(r)+1): dp[0][j] = j
    for i in range(1, len(p)+1):
        for j in range(1, len(r)+1):
            cost = 0 if p[i-1] == r[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[-1][-1] / max(1, len(r))
```

- Trainer heads‑up: during evaluation, generate with beams to compute CER on predicted text; during training, use teacher forcing (cross‑entropy loss) automatically.

---

## 19) What You Do Not Need

- You do not need the Transformers source repo for Khmer.
- You do not need to modify TrOCR model code. All Khmer specifics live in your tokenizer/processor, data, and configs.
- You do not need a detector inside Transformers; use PaddleOCR/MMOCR or simple line segmentation externally.

---

## 20) Next Actions (Immediate)

1) Stand up the environment and folders under `project_root`.
2) Create/train a minimal dataset (few thousand line crops) and JSONLs.
3) Run a baseline fine‑tune with TrOCR default tokenizer (Option A) to establish CER.
4) Build the char‑level Khmer tokenizer (Option B), resize model embeddings, retrain.
5) Integrate detector for page inference and evaluate end‑to‑end on held‑out pages.
6) Lock v1 checkpoint; start active learning loop.

---

Questions or preferences (detector choice, tokenizer option) — note them here to keep the team aligned.

