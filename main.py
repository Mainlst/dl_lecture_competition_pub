import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import optuna

from src.datasets import ThingsMEGDataset
from src.models import ResNet18
from src.utils import set_seed


def objective(trial, args):
    # ハイパーパラメータの提案
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512)

    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }

    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------
    #       Model
    # ------------------
    model = ResNet18(train_set.num_classes, train_set.num_channels).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 学習率スケジューラーの追加
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=10, verbose=True
    )

    # ------------------
    #   Start training
    # ------------------
    max_val_acc = 0
    patience = args.patience  # 早期終了のための忍耐力
    epochs_no_improve = 0  # 改善が見られなかったエポック数
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)

            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)

            with torch.no_grad():
                y_pred = model(X)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        mean_val_acc = np.mean(val_acc)
        print(
            f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {mean_val_acc:.3f}"
        )
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": np.mean(train_loss),
                    "train_acc": np.mean(train_acc),
                    "val_loss": np.mean(val_loss),
                    "val_acc": mean_val_acc,
                }
            )

        # 学習率スケジューラーのステップ
        scheduler.step(mean_val_acc)

        if mean_val_acc > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = mean_val_acc
            epochs_no_improve = 0  # 改善があったのでリセット
        else:
            epochs_no_improve += 1  # 改善がなかったエポック数を増やす

        # 早期終了のチェック
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    return max_val_acc


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials=100)

    print("Best hyperparameters: ", study.best_params)

    # 最適ハイパーパラメータでのトレーニング
    best_lr = study.best_params["lr"]
    best_batch_size = study.best_params["batch_size"]

    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    loader_args = {
        "batch_size": best_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }

    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        batch_size=best_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ResNet18(train_set.num_classes, train_set.num_channels).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=10, verbose=True
    )

    max_val_acc = 0
    patience = args.patience
    epochs_no_improve = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)

            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)

            with torch.no_grad():
                y_pred = model(X)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        mean_val_acc = np.mean(val_acc)
        print(
            f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {mean_val_acc:.3f}"
        )
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": np.mean(train_loss),
                    "train_acc": np.mean(train_acc),
                    "val_loss": np.mean(val_loss),
                    "val_acc": mean_val_acc,
                }
            )

        scheduler.step(mean_val_acc)

        if mean_val_acc > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = mean_val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(
        torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device)
    )

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Test"):
        preds.append(model(X.to(args.device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission.npy"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
