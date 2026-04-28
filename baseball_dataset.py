"""
Baseball Detection — PyTorch Dataset
------------------------------------
Reads the COCO-format splits produced by prepare_dataset.py and yields
(image_tensor, target_dict) tuples in the shape torchvision detection
models expect:

    target = {
        "boxes":    tensor[N, 4]  (xyxy, absolute pixel coords)
        "labels":   tensor[N]     (int64 class ids; 1 = baseball)
        "image_id": tensor[1]
        "area":     tensor[N]
        "iscrowd":  tensor[N]
    }

This is intentionally dependency-light: only torch, torchvision, and
PIL. No pycocotools required for training — the COCO JSON is just a
convenient on-disk format we parse directly.
"""

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class BaseballDetectionDataset(Dataset):
    """
    Parameters
    ----------
    split_dir : str
        Path to a split directory created by prepare_dataset.py. Must
        contain 'images/' and 'annotations.json'.
    transforms : callable, optional
        A function `(PIL.Image, target_dict) -> (tensor, target_dict)`.
        If None, the image is converted to a float tensor in [0, 1] with
        no augmentation.
    """

    def __init__(
        self,
        split_dir: str,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.split_dir = split_dir
        self.images_dir = os.path.join(split_dir, "images")
        self.transforms = transforms

        coco_path = os.path.join(split_dir, "annotations.json")
        if not os.path.isfile(coco_path):
            raise FileNotFoundError(f"COCO file missing: {coco_path}")
        with open(coco_path) as f:
            coco = json.load(f)

        self.images: List[dict] = coco["images"]
        # Group annotations by image_id for O(1) lookup
        self._annos_by_image: Dict[int, List[dict]] = {}
        for a in coco["annotations"]:
            self._annos_by_image.setdefault(a["image_id"], []).append(a)

        # Sanity: every image should have at least one annotation for
        # detection training (torchvision tolerates empty targets, but
        # we warn since an empty train set is almost always a bug).
        empty = [i for i in self.images if i["id"] not in self._annos_by_image]
        if empty and split_dir.rstrip("/").endswith("train"):
            print(f"[WARN] {len(empty)} train images have no annotations")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        info = self.images[idx]
        img_path = os.path.join(self.images_dir, info["file_name"])
        img = Image.open(img_path).convert("RGB")

        annos = self._annos_by_image.get(info["id"], [])

        # COCO bbox is [x, y, w, h]; torchvision wants xyxy.
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []
        for a in annos:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(int(a["category_id"]))
            areas.append(float(a.get("area", w * h)))
            iscrowd.append(int(a.get("iscrowd", 0)))

        target = {
            "boxes":    torch.as_tensor(boxes,   dtype=torch.float32).reshape(-1, 4),
            "labels":   torch.as_tensor(labels,  dtype=torch.int64),
            "image_id": torch.tensor([info["id"]], dtype=torch.int64),
            "area":     torch.as_tensor(areas,   dtype=torch.float32),
            "iscrowd":  torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = TF.to_tensor(img)

        return img, target


# -----------------------------------------------------------------
#  Minimal transform helpers (detection-aware)
# -----------------------------------------------------------------


class Compose:
    """Chain detection-aware transforms."""
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToTensor:
    """PIL image -> float tensor in [0, 1]. Target untouched."""
    def __call__(self, img, target):
        return TF.to_tensor(img), target


class RandomHorizontalFlip:
    """Flip both image and boxes with probability p."""
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img, target):
        import random
        if random.random() < self.p:
            img = TF.hflip(img)
            w = img.size[0] if hasattr(img, "size") else img.shape[-1]
            boxes = target["boxes"].clone()
            if boxes.numel():
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return img, target


def default_train_transforms() -> Compose:
    return Compose([RandomHorizontalFlip(0.5), ToTensor()])


def default_eval_transforms() -> Compose:
    return Compose([ToTensor()])


# Detection models want `list[image], list[target]` rather than stacked
# tensors, since image dimensions may vary. This collate_fn matches the
# torchvision/references pattern.
def collate_fn(batch):
    return tuple(zip(*batch))
