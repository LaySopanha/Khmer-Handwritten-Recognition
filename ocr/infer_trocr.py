import argparse
from typing import List
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


def load_model(ckpt: str):
    processor = TrOCRProcessor.from_pretrained(ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt)
    model.eval()
    return processor, model


@torch.inference_mode()
def recognize_images(processor, model, image_paths: List[str], num_beams: int = 5, max_new_tokens: int = 256) -> List[str]:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    enc = processor(images=images, return_tensors="pt")
    pixel_values = enc["pixel_values"]
    if torch.cuda.is_available():
        model = model.cuda()
        pixel_values = pixel_values.cuda()
    generated = model.generate(pixel_values=pixel_values, num_beams=num_beams, max_new_tokens=max_new_tokens)
    texts = processor.batch_decode(generated, skip_special_tokens=True)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    args = ap.parse_args()
    processor, model = load_model(args.ckpt)
    texts = recognize_images(processor, model, args.images)
    for p, t in zip(args.images, texts):
        print(f"{p}\t{t}")


if __name__ == "__main__":
    main()

