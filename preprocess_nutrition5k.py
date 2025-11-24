# preprocess_nutrition5k.py
from pathlib import Path
import os, json
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_labels(csv_file):
    labels = {}
    if csv_file is None:
        return labels
    import csv
    with open(csv_file, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row: continue
            dish_id, val = row[0], float(row[1])
            labels[dish_id] = val
    return labels

def preprocess_split(
    root_dir: str | Path,
    split: str,             # "train" or "test"
    csv_file: str | Path | None,   # labels csv for train only
    out_dir: str,
    exclude_ids=("dish_2368",),
):
    root = Path(root_dir)
    rgb_dir   = root / "Nutrition5K" / split / "color"
    depth_dir = root / "Nutrition5K" / split / "depth_raw"

    ids = sorted([p.name for p in rgb_dir.glob("*") if p.is_dir() and p.name not in exclude_ids])

    # read first image to get H,W
    sample_rgb = np.asarray(Image.open(rgb_dir / ids[0] / "rgb.png").convert("RGB"), dtype=np.uint8)
    H, W = sample_rgb.shape[:2]

    # labels
    labels_map = load_labels(csv_file)
    has_labels = csv_file is not None

    # prepare output dir & memmaps
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rgb_mm_path   = out / f"{split}_rgb_uint8.npy"
    depth_mm_path = out / f"{split}_depth_uint16.npy"
    labels_path   = out / f"{split}_labels_float32.npy"
    ids_path      = out / f"{split}_ids.txt"

    N = len(ids)
    rgb_mm   = np.memmap(rgb_mm_path, dtype=np.uint8,  mode="w+", shape=(N, H, W, 3))
    depth_mm = np.memmap(depth_mm_path, dtype=np.uint16, mode="w+", shape=(N, H, W))
    if has_labels:
        y_arr = np.memmap(labels_path, dtype=np.float32, mode="w+", shape=(N,))
    else:
        y_arr = None

    print(f"writing memmaps: N={N}, HxW={H}x{W}")
    for i, dish_id in enumerate(tqdm(ids, desc=f"Preprocessing {split}")):
        # RGB
        rgb = np.asarray(Image.open(rgb_dir / dish_id / "rgb.png").convert("RGB"), dtype=np.uint8)
        rgb_mm[i] = rgb

        # Depth (keep raw 16-bit)
        depth = np.asarray(Image.open(depth_dir / dish_id / "depth_raw.png"))
        depth_mm[i] = depth

        # Label
        if has_labels:
            y_arr[i] = float(labels_map[dish_id])

    with open(ids_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ids))

    meta = {
        "N": N, "H": H, "W": W, "split": split,
        "rgb_path": str(rgb_mm_path),
        "depth_path": str(depth_mm_path),
        "labels_path": str(labels_path) if has_labels else None,
        "ids_path": str(ids_path),
    }
    with open(out / f"{split}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    rgb_mm.flush()
    depth_mm.flush()
    if has_labels:
        y_arr.flush()

    del rgb_mm, depth_mm, y_arr

if __name__ == "__main__":
    root = Path("./")
    out  = "./Nutrition5K_memmaps"
    preprocess_split(root, "train", csv_file=root / "Nutrition5K" / "nutrition5k_train.csv", out_dir=out)
    preprocess_split(root, "test",  csv_file=None, out_dir=out)
