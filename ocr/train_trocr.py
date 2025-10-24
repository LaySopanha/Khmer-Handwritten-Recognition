import argparse
import os
import json
from dataclasses import dataclass

import torch
from transformers import (
    TrOCRProcessor,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    TrainingArguments,
    Trainer,
)

from .data import OCRJsonlDataset, OCRCollator
from .metrics import cer as cer_fn


@dataclass
class Paths:
    train_jsonl: str
    valid_jsonl: str
    image_root_train: str | None
    image_root_valid: str | None


def load_processor_and_model(model_name: str, image_height: int = 384, custom_tokenizer_dir: str | None = None):
    if custom_tokenizer_dir:
        processor = TrOCRProcessor(
            image_processor=ViTImageProcessor(size={"height": image_height}, do_resize=True, do_normalize=True),
            tokenizer=TrOCRProcessor.from_pretrained(custom_tokenizer_dir).tokenizer,
        )
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model.resize_token_embeddings(len(processor.tokenizer))
    else:
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    return processor, model


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    # replace -100 in the labels as we can't decode them
    labels_ids[labels_ids == -100] = 0
    tokenizer = compute_metrics.tokenizer  # type: ignore
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    total, dist = 0, 0.0
    for p, r in zip(pred_str, label_str):
        dist += cer_fn(p, r) * max(1, len(r))
        total += max(1, len(r))
    cer = dist / max(1, total)
    return {"cer": cer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--valid_jsonl", required=True)
    ap.add_argument("--model", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--output", default="runs/trocr-km")
    ap.add_argument("--image_height", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr_decoder", type=float, default=1e-4)
    ap.add_argument("--lr_encoder", type=float, default=1e-5)
    ap.add_argument("--max_target_length", type=int, default=256)
    ap.add_argument("--tokenizer_dir", default=None)
    args = ap.parse_args()

    processor, model = load_processor_and_model(args.model, args.image_height, args.tokenizer_dir)

    train_ds = OCRJsonlDataset(args.train_jsonl)
    valid_ds = OCRJsonlDataset(args.valid_jsonl)
    collator = OCRCollator(processor, max_target_length=args.max_target_length)

    # Parameter groups for different LRs (encoder/decoder)
    encoder = model.get_encoder()
    decoder = model.get_decoder()
    enc_params = set(p for p in encoder.parameters())
    dec_params = set(p for p in decoder.parameters())
    other_params = [p for p in model.parameters() if p not in enc_params and p not in dec_params]
    optim_groups = [
        {"params": list(dec_params), "lr": args.lr_decoder},
        {"params": list(enc_params), "lr": args.lr_encoder},
        {"params": other_params},
    ]

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=torch.cuda.is_available(),
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(None, None),
    )

    compute_metrics.tokenizer = processor.tokenizer  # type: ignore

    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    with open(os.path.join(args.output, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(training_args.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()

