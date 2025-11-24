import argparse
import os
import csv
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from model import RGBDFusionNet, RGBResNetCalories, RGBDResNetLateFusion, RGBDFusionCAB
from model import RGBDCaloryPredictor
from dataset_nutrition import Nutrition5KDataset, make_train_val_dataloaders, Nutrition5KMemmap, make_mmap_train_val_loaders
from trainer import Trainer
import torchvision.transforms as T

def count_nonzero_colary(csv_path: str):
    df = pd.read_csv(csv_path)
    values = df["Value"]
    n_zero = (values <= 0).sum()
    n_nonzero = (values > 0).sum()
    return int(n_zero), int(n_nonzero)

def build_transforms(use_transforms: bool = True):
    if not use_transforms:
        return None, None

    rgb_tf = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    depth_tf = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    return rgb_tf, depth_tf


def make_test_loader(root_dir, rgb_tf, depth_tf, batch_size, num_workers):
    depth_stats_path = Path(os.path.join(root_dir, 'depth_stats.json'))
    with depth_stats_path.open("r") as f:
        depth_stats = json.load(f)

    test_ds = Nutrition5KDataset(
        root_dir=root_dir,
        split="test",
        rgb_transform=rgb_tf,
        depth_transform=depth_tf,
        return_id=True,
        use_depth_color=False,
        depth_stats=depth_stats,
    )
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def make_mmap_test_loader(meta_path, batch_size, num_workers):
    test_ds = Nutrition5KMemmap(
        meta_path,
    )
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

def main():
    parser = argparse.ArgumentParser(description="RGB-D Nutrition5K Trainer")
    parser.add_argument("--data_root", type=str, required=True, help="Path to Nutrition5K root")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train CSV")

    # trainer
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--multi_lambda", type=float, default=0.1)

    # data preprocessing
    parser.add_argument("--use_transform", action="store_true",)

    # model
    parser.add_argument("--model", type=str, default="RGBDFusionNet")
    parser.add_argument("--fpn_out_ch", type=int, default=256)
    parser.add_argument("--spatial_dropout_ratio", type=float, default=0.0)
    parser.add_argument("--head_dropout_ratio", type=float, default=0.3)

    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--test_freq", type=int, default=30, help="Validate every N steps")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--project_name", type=str, default="Nutrition5k Prediction")
    parser.add_argument("--pred_out_dir", type=str, default="test_predictions", help="Where to save test predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_ensembling", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = (
        f"{args.model}_"
        f"bs{args.batch_size}_"
        f"lr{args.lr}_"
        f"tf{args.use_transform}_"
        f"fpn{args.fpn_out_ch}_"
        f"sd{args.spatial_dropout_ratio}_"
        f"hd{args.head_dropout_ratio}"
    )

    rgb_tf, depth_tf = build_transforms(args.use_transform)

    # train_loader_full, val_loader_full = make_train_val_dataloaders(
    #     root_dir=args.data_root, csv_file=args.train_csv,
    #     rgb_transform=rgb_tf, depth_transform=depth_tf,
    #     batch_size=args.batch_size, val_ratio=0.05, num_workers=args.num_workers,
    #     non_zero_colary=False, target_log1p=False
    # )

    root = Path(args.data_root)
    train_meta_path = root / "Nutrition5K_memmaps" / "train_meta.json"
    test_meta_path = root / "Nutrition5K_memmaps" / "test_meta.json"

    train_loader_full, val_loader_full = make_mmap_train_val_loaders(
        meta_path=train_meta_path, batch_size=args.batch_size, val_ratio=0.05, num_workers=args.num_workers)

    # train_loader_non_zero_colary, val_loader_non_zero_colary = make_train_val_dataloaders(
    #     root_dir=args.data_root, csv_file=args.train_csv,
    #     rgb_transform=rgb_tf, depth_transform=depth_tf,
    #     batch_size=args.batch_size, val_ratio=0.05, num_workers=args.num_workers,
    #     non_zero_colary=True, target_log1p=True
    # )

    # test_loader = make_test_loader(args.data_root, rgb_tf, depth_tf, args.batch_size, args.num_workers)
    test_loader = make_mmap_test_loader(test_meta_path, batch_size=args.batch_size, num_workers=args.num_workers)

    # load depth stats
    depth_stats_path = Path(os.path.join(args.data_root, 'depth_stats.json'))
    with depth_stats_path.open("r") as f:
        depth_stats = json.load(f)

    assert args.model in ["RGBDFusionNet", "RGBDFusionCAB", "RGBDResNetLateFusion", "RGBResNetCalories", "ensemble"]

    if not args.enable_ensembling:
        # backbone selection
        if args.model == "RGBDFusionNet":
            backbone = RGBDFusionNet(
                fpn_out_ch=args.fpn_out_ch,
                spatial_dropout=args.spatial_dropout_ratio,
            )
            in_features = backbone.head_in
        elif args.model == "RGBDResNetLateFusion":
            backbone = RGBDResNetLateFusion()
            in_features = backbone.head_in
        elif args.model == "RGBDFusionCAB":
            backbone = RGBDFusionCAB()
            in_features = backbone.head_in
        else:  # "RGBResNet18"
            backbone = RGBResNetCalories()
            in_features = backbone.head_in

        model = RGBDCaloryPredictor(backbone, arch_name=args.model, device=device,
                                    in_features=in_features, hidden=256, p=args.head_dropout_ratio).to(device)

        # # Stage A: train binary classification on non-zero colary
        # n_zero, n_pos = count_nonzero_colary(args.train_csv)
        # pos_weight = (n_zero / max(1, n_pos))  # positive label is y>0
        # opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # trainer = Trainer(model, args.model, pos_weight, args.multi_lambda, opt, device,
        #                   project_name=args.project_name, experiment_name=experiment_name)
        # trainer.fit(train_loader_full, val_loader_full, args.epochs, "bc")

        # # reload the best ckpt from Stage A
        # trainer.load_checkpoint()

        # Stage B: train regression on non-zero colary (frozen backbone)
        # opt = AdamW(model.value_head.parameters(), lr=args.lr, weight_decay=1e-4)
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        trainer = Trainer(model, args.model, 0.0, args.multi_lambda, opt, device,
                          project_name=args.project_name, experiment_name=experiment_name, depth_stat=depth_stats)
        # trainer.fit(train_loader_non_zero_colary, val_loader_non_zero_colary, args.epochs, "value")
        trainer.fit(train_loader_full, val_loader_full, args.epochs, "value")

        # reload the best ckpt from Stage B
        trainer.load_checkpoint()

        # # Stage C: Multitask (unfreeze, small lr)
        # opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # trainer = Trainer(model, args.model, pos_weight, args.multi_lambda, opt, device,
        #                   project_name=args.project_name, experiment_name=experiment_name)
        # trainer.fit(train_loader_full, val_loader_full, args.epochs, "multi")
        #
        # # reload the best ckpt from Stage C
        # trainer.load_checkpoint()

        # inference and save predictions
        exp_dir = os.path.join(args.pred_out_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        pred_path = os.path.join(exp_dir, "predictions.csv")

        # Save predictions
        trainer.save_predict_to_file(test_loader, pred_path)
        print(f"Saved test predictions to: {pred_path}")

    else: # enabling ensembling
        fn_backbone = RGBDFusionNet(
            fpn_out_ch=args.fpn_out_ch,
            spatial_dropout=args.spatial_dropout_ratio,
        )
        fn_in_features = fn_backbone.head_in

        fcab_backbone = RGBDFusionCAB()
        fcab_in_features = fcab_backbone.head_in

        # model
        fn_model = RGBDCaloryPredictor(fn_backbone, arch_name="RGBDFusionNet", device=device,
                                       in_features=fn_in_features, hidden=256, p=args.head_dropout_ratio).to(device)
        fcab_model = RGBDCaloryPredictor(fcab_backbone, arch_name="RGBDFusionCAB", device=device,
                                         in_features=fcab_in_features, hidden=256, p=args.head_dropout_ratio).to(device)

        # trainer
        print(f"Start training RGBDFusionNet...")
        fn_opt = AdamW(fn_model.parameters(), lr=args.lr, weight_decay=1e-4)
        fn_trainer = Trainer(fn_model, "RGBDFusionNet", 0.0, args.multi_lambda, fn_opt, device,
                             project_name=args.project_name, experiment_name=experiment_name, depth_stat=depth_stats)
        fn_trainer.fit(train_loader_full, val_loader_full, args.epochs, "value")
        if fn_trainer.wandb: fn_trainer.wandb.finish()
        fn_trainer.load_checkpoint()
        print(f"Best validation score from {fn_trainer.arch_name}: {fn_trainer.best_val}")

        print(f"Start training RGBDFusionCAB...")
        fcab_opt = AdamW(fcab_model.parameters(), lr=args.lr, weight_decay=1e-4)
        fcab_trainer = Trainer(fcab_model, "RGBDFusionCAB", 0.0, args.multi_lambda, fcab_opt, device,
                               project_name=args.project_name, experiment_name=experiment_name, depth_stat=depth_stats)
        fcab_trainer.fit(train_loader_full, val_loader_full, args.epochs, "value")
        if fcab_trainer.wandb: fcab_trainer.wandb.finish()
        fcab_trainer.load_checkpoint()
        print(f"Best validation score from {fcab_trainer.arch_name}: {fcab_trainer.best_val}")

        # ensembling
        weights = eval_ens_model_complementarity(fn_model, fcab_model, val_loader_full, device, depth_stats, out_dir="results")
        preds = predict_colary_ensemble([fn_model, fcab_model], weights, test_loader, device, depth_stats, mode="test")

        exp_dir = os.path.join(args.pred_out_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        pred_path = os.path.join(exp_dir, "predictions.csv")

        with open(pred_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Value"])
            for i, p in zip(preds["ids"], preds["pred"]):
                writer.writerow([i, f"{p:.6f}"])

@torch.no_grad()
def predict_colary_single_model(model, loader, device, depth_stats):
    model.eval()
    ys, yhat, ids = [], [], []
    for batch in loader:
        # preprocessing
        rgb   = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        y     = batch["y"].to(device) # [B,]
        batch["rgb"]   = rgb / 255.0
        batch["depth"] = (depth - depth_stats["min"]) / (depth_stats["max"] - depth_stats["min"])
        batch["target"] = y

        pred = model.predict_colary_value(batch)   # [B,1]
        ys.append(y)
        yhat.append(pred)
        ids.extend(batch.get("id", [None] * pred.size(0)))

    y = torch.cat(ys, dim=0) # [N]
    p = torch.cat(yhat, dim=0).view(-1) # [N]
    return {"y": y.cpu(), "pred": p.cpu(), "ids": ids}

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

def plot_residual_scatter(resA, resB, out_path, nameA="A", nameB="B"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.scatter(resA, resB, s=6, alpha=0.5)
    plt.axhline(0, lw=1); plt.axvline(0, lw=1)
    plt.xlabel(f"{nameA} residual"); plt.ylabel(f"{nameB} residual")
    plt.title("Residual scatter (single task)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def calculate_ensemble_weights(mseA, mseB, eps=1e-8):
    wA = 1.0 / (mseA + eps)
    wB = 1.0 / (mseB + eps)
    s = wA + wB
    return float(wA/s), float(wB/s)

def eval_ens_model_complementarity(fn_model, fcab_model, val_loader, device, depth_stats, out_dir="results"):
    A = predict_colary_single_model(fn_model, val_loader, device, depth_stats)
    B = predict_colary_single_model(fcab_model, val_loader, device, depth_stats)

    y  = A["y"] # [N,]
    pA = A["pred"]; pB = B["pred"] # [N,]

    crition_mse = nn.MSELoss()
    mseA = crition_mse(pA, y).item()
    mseB = crition_mse(pB, y).item()

    resA = (y - pA).numpy()
    resB = (y - pB).numpy()
    corr = float(np.corrcoef(resA, resB)[0, 1])

    plot_residual_scatter(resA, resB,
                          os.path.join(out_dir, "residual_scatter.png"),
                          "RGBDFusionNet", "RGBDFusionCAB")

    wA, wB = calculate_ensemble_weights(mseA, mseB)

    AB = predict_colary_ensemble(models=[fn_model, fcab_model], weights=[wA, wB],
                                 loader=val_loader, device=device, depth_stats=depth_stats)
    pAB = AB["pred"]
    mseAB = crition_mse(pAB, y).item()

    print(f"MSE A={mseA:.4f}  B={mseB:.4f}  Ens={mseAB:.4f}  resid_corr={corr:+.3f}")
    print(f"Weights: A={wA:.3f}, B={wB:.3f}")

    # save CSV
    df = pd.DataFrame({
        "id": A["ids"],
        "y":  y.numpy(),
        "A":  pA.numpy(),
        "B":  pB.numpy(),
        "AB": pAB.numpy(),
    })
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "val_preds.csv"), index=False)

    return wA, wB

if __name__ == "__main__":
    main()

