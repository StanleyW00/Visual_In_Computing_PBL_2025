#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm

# ===== 可选：AUC =====
try:
    from sklearn.metrics import roc_auc_score
    HAS_SK = True
except Exception:
    HAS_SK = False

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# --------------------- utils ---------------------
def set_seed(seed=42):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_transforms(img_size=224, eval_resize=256):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(eval_resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return train_tf, val_tf

def build_dataloaders(root, batch_size=32, num_workers=4,
                      img_size=224, eval_resize=256,
                      sampler_mode="none"):
    """
    sampler_mode: "none" | "weighted_sampler" | "class_weight"
    """
    train_tf, val_tf = build_transforms(img_size, eval_resize)
    train_ds = datasets.ImageFolder(str(Path(root) / "Train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(Path(root) / "Test"),  transform=val_tf)

    # 统计类别
    counts = Counter(train_ds.targets)
    num_classes = len(train_ds.classes)
    print(f"[INFO] #classes={num_classes}  train={len(train_ds)}  val={len(val_ds)}")
    print("[INFO] Classes:", train_ds.classes)

    # 采样器 or 类权重
    train_sampler = None
    class_weights_tensor = None

    if sampler_mode == "weighted_sampler":
        # 每个样本的权重 = 1 / 该样本类别计数
        sample_weights = [1.0 / counts[y] for y in train_ds.targets]
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
        print("[INFO] Using WeightedRandomSampler (balanced sampling).")
    else:
        shuffle = True
        if sampler_mode == "class_weight":
            # 类权重（按 1/count）
            weights = torch.tensor([1.0 / counts[c] for c in range(num_classes)], dtype=torch.float32)
            class_weights_tensor = weights
            print("[INFO] Using class-weighted CrossEntropyLoss.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=shuffle if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    return train_ds, val_ds, train_loader, val_loader, class_weights_tensor

class MAEClassifier(nn.Module):
    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)  # [B, feat]
        return self.classifier(x)

def create_model(model_name: str, num_classes: int, device):
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)  # feature extractor
    model = MAEClassifier(backbone, num_classes=num_classes).to(device)
    return model

def freeze_backbone(model: MAEClassifier, freeze=True):
    for p in model.backbone.parameters():
        p.requires_grad = not (freeze)
    for p in model.classifier.parameters():
        p.requires_grad = True

def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    all_probs = []
    all_labels = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)

            probs = softmax(logits)
            preds = torch.argmax(probs, dim=1)
            total_correct += (preds == labels).sum().item()
            total += images.size(0)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / total if criterion is not None else 0.0
    acc = total_correct / total

    # Multi-class AUC (macro, OvR)
    auc = None
    if HAS_SK:
        try:
            import numpy as np
            y_true = torch.cat(all_labels, dim=0).numpy()
            y_score = torch.cat(all_probs, dim=0).numpy()
            # 如果验证集中有的类别没出现，roc_auc_score 可能会报错；交给 try
            auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        except Exception:
            auc = None
    return avg_loss, acc, auc

def train_one_epoch(model, loader, device, optimizer, criterion, use_amp=False):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    if (use_amp and device.type == "cuda"):
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, total_correct / total

def save_ckpt(model, path):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved: {path}")

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ISIC_dataset", help="dataset root containing Train/Test")
    ap.add_argument("--model", default="vit_base_patch16_224.mae")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--eval-resize", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    # 采样/权重
    ap.add_argument("--balance", choices=["none", "weighted_sampler", "class_weight"], default="none")

    # 两阶段
    ap.add_argument("--probe-epochs", type=int, default=5, help="linear probe epochs (freeze backbone)")
    ap.add_argument("--ft-epochs", type=int, default=10, help="finetune epochs (unfreeze all)")

    # 优化器
    ap.add_argument("--probe-lr", type=float, default=3e-4)
    ap.add_argument("--ft-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)

    # 其他
    ap.add_argument("--amp", action="store_true", help="enable AMP on CUDA")
    ap.add_argument("--outdir", default="checkpoints")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("[INFO] Device:", device)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # data
    train_ds, val_ds, train_loader, val_loader, class_weights = build_dataloaders(
        root=args.root, batch_size=args.batch_size, num_workers=args.num_workers,
        img_size=args.img_size, eval_resize=args.eval_resize, sampler_mode=args.balance
    )
    num_classes = len(train_ds.classes)

    # model
    model = create_model(args.model, num_classes=num_classes, device=device)

    # loss
    if args.balance == "class_weight" and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # ===== Stage 1: Linear Probe (freeze backbone) =====
    if args.probe_epochs > 0:
        print("\n===== Stage 1: Linear Probe (freeze backbone) =====")
        freeze_backbone(model, freeze=True)
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.probe_lr, weight_decay=args.weight_decay)
        best_score = -1.0
        best_path = str(Path(args.outdir) / "best_probe.pt")

        for epoch in range(1, args.probe_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion, use_amp=args.amp)
            val_loss, val_acc, val_auc = evaluate(model, val_loader, device, criterion)
            key_metric = val_auc if (val_auc is not None) else val_acc

            print(f"[Probe][{epoch}/{args.probe_epochs}] "
                  f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc if val_auc is not None else 'NA'}")

            if key_metric > best_score:
                best_score = key_metric
                save_ckpt(model, best_path)

        print(f"[INFO] Best Probe Metric={best_score:.4f} (saved to {best_path})")

        # （可选）加载最优探针权重再进入微调
        model.load_state_dict(torch.load(best_path, map_location=device))

    # ===== Stage 2: Finetune (unfreeze all) =====
    if args.ft_epochs > 0:
        print("\n===== Stage 2: Finetune (unfreeze all) =====")
        freeze_backbone(model, freeze=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=args.weight_decay)

        best_score = -1.0
        best_path = str(Path(args.outdir) / "best_finetune.pt")

        for epoch in range(1, args.ft_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion, use_amp=args.amp)
            val_loss, val_acc, val_auc = evaluate(model, val_loader, device, criterion)
            key_metric = val_auc if (val_auc is not None) else val_acc

            print(f"[FT][{epoch}/{args.ft_epochs}] "
                  f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc if val_auc is not None else 'NA'}")

            # 保存最好
            if key_metric > best_score:
                best_score = key_metric
                save_ckpt(model, best_path)

            # 也保存 last
            save_ckpt(model, str(Path(args.outdir) / "last_finetune.pt"))

        print(f"[INFO] Best Finetune Metric={best_score:.4f} (saved to {best_path})")

    print("\n[DONE]")

if __name__ == "__main__":
    main()
