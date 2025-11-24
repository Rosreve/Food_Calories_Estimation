import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import os
from torch.nn import BCEWithLogitsLoss, MSELoss

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        arch_name: str,
        pos_weight: float,
        multitask_lambda: float,  # for multi-task weighting
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_freq: int = 10,
        val_freq: int = 50,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        depth_stat: dict = None
    ):
        self.model = model.to(device)
        self.arch_name = arch_name
        self.optimizer = optimizer
        self.device = device
        self.criterion_l1 = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()
        self.criterion_mse_ps = nn.MSELoss(reduction='none')
        self.criterion_bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        self.multitask_lambda = multitask_lambda
        self.log_freq = log_freq
        self.val_freq = val_freq
        self.global_step = 1

        self.best_val = float("inf")
        self.ckpt_dir = os.path.join(checkpoint_dir, experiment_name, arch_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_ckpt_path = os.path.join(self.ckpt_dir, "best.pt")
        self.depth_stat = depth_stat

        self.wandb = None
        if project_name:
            import wandb
            self.wandb = wandb
            wandb.init(project=project_name, name=experiment_name, config={})

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), self.best_ckpt_path)

    def load_checkpoint(self):
        state = torch.load(self.best_ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)

    def _batch_preprocssing(self, batch):
        rgb = batch["rgb"].to(self.device)
        depth = batch["depth"].to(self.device)
        y = batch["y"].to(self.device)
        batch["rgb"] = rgb / 255.0
        batch["depth"] = (depth - self.depth_stat["min"]) / (self.depth_stat["max"] - self.depth_stat["min"])
        # batch["target"] = torch.log1p(y)
        batch["target"] = y
        batch["zero_colary"] = (y == 0).float()
        return batch

    def _step(self, batch, task: str, train: bool = True): # task: ["bc", "value", "multi"]
        self.model.train(train)
        batch = self._batch_preprocssing(batch)
        y = batch["target"].view(-1, 1)
        z = batch["zero_colary"].view(-1, 1)

        if task == "bc":
            preds_bc = self.model.predict_zero_colary(batch)
            loss = self.criterion_bce(preds_bc, z)
        elif task == "value":
            preds_v = self.model.predict_colary_value(batch)
            #loss = self.criterion_mse(preds_v, y)
            loss = self.criterion_l1(preds_v, y)

            # if self.global_step % 100 == 0:  # print every 100 steps
            #     print("\n[DEBUG step]", self.global_step)
            #     print("preds_v:", preds_v.detach().cpu().flatten()[:10])
            #     print("target (y):", y.detach().cpu().flatten()[:10])
            #     print("loss:", loss.item())
            #     print("----------------------")
        else: # multi
            preds_bc, preds_v = self.model(batch)
            loss_bc = self.criterion_bce(preds_bc, z)

            pos_mask = (y > 0).float()  # 1 where we have calorie ground truth > 0

            se = self.criterion_mse_ps(preds_v, y)  # shape (B,1)
            mse_pos = (se * pos_mask).sum() / pos_mask.sum().clamp_min(1.0)
            loss = self.multitask_lambda * loss_bc + (1.0 - self.multitask_lambda) * mse_pos

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return loss.detach(), y.detach()

    @torch.no_grad()
    def _validate(self, task: str, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            batch = self._batch_preprocssing(batch)
            y = batch["target"].view(-1, 1)
            z = batch["zero_colary"].view(-1, 1)

            if task == "bc":
                preds_bc = self.model.predict_zero_colary(batch)
                loss = self.criterion_bce(preds_bc, z)
            elif task == "value":
                preds_v = self.model.predict_colary_value(batch)
                loss = self.criterion_mse(preds_v, y)
            else:  # multi
                preds_bc, preds_v = self.model(batch)
                loss_bc = self.criterion_bce(preds_bc, z)

                pos_mask = (y > 0).float()  # 1 where we have calorie ground truth > 0

                se = self.criterion_mse_ps(preds_v, y)  # shape (B,1)
                mse_pos = (se * pos_mask).sum() / pos_mask.sum().clamp_min(1.0)
                loss = self.multitask_lambda * loss_bc + (1.0 - self.multitask_lambda) * mse_pos

            bs = y.numel()
            total_loss += loss.item() * bs
            total_count += bs

        return total_loss / max(1, total_count)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, task: str):
        for epoch in tqdm(range(1, epochs + 1), desc=f"Training ({task})", unit="epoch"):
            running_loss = 0.0
            running_count = 0

            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, unit="batch")

            for batch in batch_iter:
                loss, y = self._step(batch, task, train=True)

                bs = y.numel()
                running_loss += loss.item() * bs
                running_count += bs

                if self.global_step % self.log_freq == 0:
                    train_loss = running_loss / max(1, running_count)
                    if self.wandb:
                        self.wandb.log({f"train/{self.arch_name}/{task}/loss": train_loss}, step=self.global_step)
                    running_loss = 0.0
                    running_count = 0

                if self.global_step % self.val_freq == 0:
                    val_loss = self._validate(task, val_loader)
                    if self.wandb:
                        self.wandb.log({f"val/{self.arch_name}/{task}/loss": val_loss}, step=self.global_step)

                    if val_loss < self.best_val:
                        self.best_val = val_loss
                        self._save_checkpoint()

                self.global_step += 1

    @torch.no_grad()
    def save_predict_to_file(self, test_loader: DataLoader, out_path: str):
        self.model.eval()
        import csv
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Value"])
            for batch in test_loader:
                rgb = batch["rgb"].to(self.device)
                depth = batch["depth"].to(self.device)
                batch["rgb"] = rgb / 255.0
                batch["depth"] = (depth - self.depth_stat["min"]) / (self.depth_stat["max"] - self.depth_stat["min"])

                sid = batch["id"]
                preds = self.model.predict(batch).squeeze(-1).cpu().tolist()

                for i, p in zip(sid, preds):
                    writer.writerow([i, f"{p:.6f}"])

