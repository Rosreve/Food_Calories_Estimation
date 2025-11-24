import os
import csv
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import RGBDFusionCAB
from model import RGBDCaloryPredictor
from dataset_nutrition import Nutrition5KDataset


def make_test_loader(root_dir, rgb_tf, depth_tf, batch_size, num_workers):
    test_ds = Nutrition5KDataset(
        root_dir=root_dir,
        split="test",
    )
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def list_checkpoints(dir_path: str) -> list[Path]:
    p = Path(dir_path)
    files = sorted(list(p.glob("*.pt")))
    return files


def load_state(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)


def build_fcab_model(device: torch.device) -> nn.Module:
    # Backbone
    fcab_backbone = RGBDFusionCAB()
    fcab_in_features = fcab_backbone.head_in

    # Predictor wrapper
    model = RGBDCaloryPredictor(
        backbone=fcab_backbone,
        arch_name="RGBDFusionCAB",
        device=device,
        in_features=fcab_in_features,
        hidden=256,
        p=0.3,
    ).to(device)
    return model


@torch.no_grad()
def run_and_save(models: list[nn.Module],
                 test_loader,
                 device: torch.device,
                 depth_stats: dict[str, float],
                 out_dir: Path):
    # Ensemble predict (uniform weights if None)
    preds = predict_colary_ensemble(
        models=models,
        weights=None,
        loader=test_loader,
        device=device,
        depth_stats=depth_stats,
        mode="test",
    )

    pred_path = out_dir / "fusioncab_ensemble_predictions.csv"
    os.makedirs(pred_path.parent, exist_ok=True)

    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Value"])
        for i, p in zip(preds["ids"], preds["pred"]):
            writer.writerow([i, f"{float(p):.6f}"])

    print(f"Saved predictions -> {pred_path}")

@torch.no_grad()
def predict_colary_ensemble(models: list[nn.Module], weights: list[float] | None, loader, device, depth_stats, mode="val"):
    for model in models:
        model.eval()
    ys, yhat, ids = [], [], []

    M = len(models)
    if not weights:
        w = torch.full((M,), 1.0 / M, device=device, dtype=torch.float32)
    else:
        w = torch.tensor(weights, device=device, dtype=torch.float32)

    for batch in loader:
        # preprocessing
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)

        batch["rgb"] = rgb / 255.0
        batch["depth"] = (depth - depth_stats["min"]) / (depth_stats["max"] - depth_stats["min"])


        if mode == "val":
            y = batch["y"].to(device)  # [B,]
            batch["target"] = y
            ys.append(y)

        preds = []
        for model in models:
            preds.append(model.predict_colary_value(batch).squeeze(-1))
        colaries = torch.stack(preds, dim=0) # [M, B]
        pred = (w.view(-1, 1) * colaries).sum(0) # [B]

        yhat.append(pred)
        ids.extend(batch.get("id", [None] * pred.size(0)))

    p = torch.cat(yhat, dim=0)  # [N]

    if mode == "val":
        y = torch.cat(ys, dim=0)  # [N]
        return {"y": y.cpu(), "pred": p.cpu(), "ids": ids}
    else:
        return {"pred": p.cpu(), "ids": ids}


def main():
    model_pth = "checkpoints/RGBDFusionCAB_bs32_lr5e-05_tfFalse_fpn256_sd0.1_hd0.3/RGBDFusionCAB"
    test_loader = make_test_loader("./", None, None, batch_size=32, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth_stats_path = Path("./") / "depth_stats.json"
    with depth_stats_path.open("r") as f:
        depth_stats = json.load(f)

    ckpt_files = list_checkpoints(model_pth)

    # build a fresh model per checkpoint and load weights
    models: list[nn.Module] = []
    for ck in ckpt_files:
        m = build_fcab_model(device)
        load_state(m, ck, device)
        m.eval()
        models.append(m)

    # experiment name derived from directory + checkpoint stems
    our_dir = Path("test_predictions")
    run_and_save(models, test_loader, device, depth_stats, our_dir)


if __name__ == "__main__":
    main()

