import json
from typing import Any, Dict, List, Optional
from PIL import Image
from torch.utils.data import Dataset


class OCRJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: Optional[str] = None, transform=None):
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        obj = self.items[idx]
        img_path = obj["image_path"]
        if self.image_root:
            from os.path import join

            img_path = join(self.image_root, img_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = obj.get("text", "")
        return {"image": image, "text": text}


class OCRCollator:
    def __init__(self, processor, max_target_length: int = 256):
        self.processor = processor
        self.max_target_length = max_target_length

    def __call__(self, batch: List[Dict[str, Any]]):
        images = [x["image"] for x in batch]
        texts = [x["text"] for x in batch]
        enc = self.processor(
            images=images,
            text=texts,
            padding="longest",
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        pad_id = self.processor.tokenizer.pad_token_id
        labels = enc["labels"]
        labels[labels == (pad_id if pad_id is not None else -100)] = -100
        return {"pixel_values": enc["pixel_values"], "labels": labels}

