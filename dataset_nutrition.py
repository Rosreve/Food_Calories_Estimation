from pathlib import Path
from typing import Callable, Optional, Tuple, Any
import os
import json
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class Nutrition5KMemmap(Dataset):
    def __init__(
        self,
        meta_path: str | Path,
        return_id: bool = True,
    ):
        meta = json.loads(Path(meta_path).read_text())
        self.N, self.H, self.W = meta["N"], meta["H"], meta["W"]
        self.return_id = return_id

        # open memmaps in read-only mode
        self.rgb_mm   = np.memmap(meta["rgb_path"],   dtype=np.uint8,  mode="r", shape=(self.N, self.H, self.W, 3))
        self.depth_mm = np.memmap(meta["depth_path"], dtype=np.uint16, mode="r", shape=(self.N, self.H, self.W))

        # labels & ids
        self.has_labels = meta["labels_path"] is not None
        self.labels = None
        if self.has_labels:
            self.labels = np.memmap(meta["labels_path"], dtype=np.float32, mode="r", shape=(self.N,))
        self.ids = [line.strip() for line in Path(meta["ids_path"]).read_text().splitlines()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        rgb   = torch.from_numpy(self.rgb_mm[i].copy()).permute(2,0,1).float()           # [3,H,W] float32
        depth = torch.from_numpy(self.depth_mm[i].copy()).unsqueeze(0).float()           # [1,H,W] float32

        sample = {"rgb": rgb, "depth": depth}

        if self.has_labels:
            y = float(self.labels[i])
            sample["y"] = torch.tensor(y, dtype=torch.float32) # raw target

        if self.return_id:
            sample["id"] = self.ids[i]
        return sample

class Nutrition5KDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,  # train / test
        csv_file: Optional[str] = None,  # labels csv
        rgb_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        depth_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_id: bool = True,
        use_depth_color: bool = False,   # keep False for training (colorized map used only for visualization)
        depth_stats: dict[str, float] = None, # global min / max depth
        non_zero_colary: bool = False,
        target_log1p: bool = False
    ):
        # depth_stats_existed = False if depth_stats is None else True
        #
        # if depth_stats is None:
        #     depth_stats = {"min": float("inf"), "max": float("-inf")}

        self.non_zero_colary = non_zero_colary
        self.target_log1p = target_log1p

        self.root = Path(root_dir)
        self.split = split
        self.csv_file = csv_file
        self.rgb_dir   = self.root / "Nutrition5K" / split / "color"
        self.depth_raw_dir   = self.root / "Nutrition5K" / split / "depth_raw"
        self.depth_color_dir = self.root / "Nutrition5K" / split / "depth_color"

        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        self.return_id = return_id
        self.use_depth_color = use_depth_color

        self.depth_stats = depth_stats

        self.ids = sorted([p.name for p in self.rgb_dir.glob("*") if p.name not in ["dish_2368"]])
        #all_ids = sorted([p.name for p in self.rgb_dir.glob("*")])

        # self.ids = []
        # for dish_id in all_ids:
        #     rgb_path = self.rgb_dir / dish_id / "rgb.png"
        #     depth_path = self.depth_raw_dir / dish_id / "depth_raw.png"
        #     if self.is_rgb_image_safe(rgb_path) and self.is_depth_image_safe(depth_path, depth_stats_existed):
        #         self.ids.append(dish_id)
        #     else:
        #         print(f"Skipped {dish_id}: image not safe to open")

        self.labels: dict[str, float] = {}
        if csv_file is not None and Path(csv_file).exists():
            import csv
            with open(csv_file, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if not row:
                        continue
                    dish_id = row[0]
                    value = float(row[1])
                    self.labels[dish_id] = value

            if self.non_zero_colary:
                self.ids = [i for i in self.ids if self.labels.get(i, 0.0) > 0.0]

        # if not depth_stats_existed:
        #     import json
        #     save_path = self.root / "depth_stats.json"
        #     with open(save_path, "w") as f:
        #         json.dump(self.depth_stats, f, indent=2)
        #     print(f"Depth stats saved to {save_path}")

    def _read_rgb(self, path: Path) -> np.ndarray:
        """Read RGB png -> float32 [H,W,3] in [0,1]."""
        img = Image.open(path).convert("RGB")
        # arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.asarray(img, dtype=np.float32)
        return arr

    def _read_depth_raw(self, path: Path) -> np.ndarray:
        """
        Read raw depth png > float32 [H,W] normalized to [0,1] per image.
        """
        img = Image.open(path)  # keeps original bit depth (mode 'I;16')
        arr = np.asarray(img).astype(np.float32)
        #arr = (arr - self.depth_stats["min"]) / (self.depth_stats["max"] - self.depth_stats["min"])
        return arr

    def is_rgb_image_safe(self, path: str | Path) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False

    def is_depth_image_safe(self, path: str | Path, depth_stats_existed: bool) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()

            if not depth_stats_existed:
                with Image.open(path) as img:
                    depth = np.array(img, dtype=np.float32)

                depth_min = np.min(depth)
                depth_max = np.max(depth)

                self.depth_stats["min"] = float(min(self.depth_stats["min"], depth_min))
                self.depth_stats["max"] = float(max(self.depth_stats["max"], depth_max))

            return True

        except (OSError, Image.UnidentifiedImageError):
            return False

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dish_id = self.ids[idx]
        rgb_path   = self.rgb_dir / dish_id / "rgb.png"
        depth_path = self.depth_raw_dir / dish_id / "depth_raw.png"
        rgb   = self._read_rgb(rgb_path)            # [H,W,3] in [0,1]
        depth = self._read_depth_raw(depth_path)    # [H,W] in [0,1]

        # Optional: load colorized depth for visualization
        # depth_color = None
        # if self.use_depth_color:
        #     dcp = self.depth_color_dir / dish_id / "depth_color.png"
        #     if dcp.exists():
        #         depth_color = np.asarray(Image.open(dcp).convert("RGB"), dtype=np.float32) / 255.0

        # to torch tensors (C,H,W)
        rgb_t   = torch.from_numpy(rgb).permute(2, 0, 1)              # [3,H,W]
        depth_t = torch.from_numpy(depth)[None, ...]                  # [1,H,W]
        sample: dict[str, Any] = {"rgb": rgb_t, "depth": depth_t}

        # if self.use_depth_color and depth_color is not None:
        #     sample["depth_color"] = torch.from_numpy(depth_color).permute(2, 0, 1)  # [3,H,W]

        # optional per-modality transforms
        # if self.rgb_transform is not None:
        #     sample["rgb"] = self.rgb_transform(sample["rgb"])
        # if self.depth_transform is not None:
        #     sample["depth"] = self.depth_transform(sample["depth"])

        # target if available
        if self.csv_file is not None:
            y = self.labels[dish_id]
            sample["y"] = torch.tensor(y, dtype=torch.float32)
            # if self.target_log1p:
            #     y = np.log1p(y)
            # sample["target"] = torch.tensor(y, dtype=torch.float32)
            # sample["zero_colary"] = torch.tensor(1.0 if self.labels[dish_id] == 0.0 else 0.0, dtype=torch.float32)

        if self.return_id:
            sample["id"] = dish_id

        return sample


def make_mmap_train_val_loaders(
    meta_path: str | Path,
    batch_size: int = 16,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    full_train_ds = Nutrition5KMemmap(
        meta_path=meta_path,
        return_id=True
    )

    total_len = len(full_train_ds)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_train_ds, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def make_train_val_dataloaders(
    root_dir: str,
    csv_file: str,
    rgb_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    depth_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    batch_size: int = 16,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    non_zero_colary: bool = False,
    target_log1p: bool = False
):
    # assume depth stats precompted
    depth_stats_path = Path(os.path.join(root_dir, 'depth_stats.json'))
    with depth_stats_path.open("r") as f:
        depth_stats = json.load(f)

    full_train_ds = Nutrition5KDataset(
        root_dir=root_dir,
        split="train",
        csv_file=csv_file,
        rgb_transform=rgb_transform,
        depth_transform=depth_transform,
        return_id=True,
        use_depth_color=False,
        depth_stats=depth_stats,
        non_zero_colary=non_zero_colary,
        target_log1p=target_log1p,
    )

    total_len = len(full_train_ds)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_train_ds, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
