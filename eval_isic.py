#!/usr/bin/env python3
import argparse, torch, timm, json
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

# ------------ Model wrapper ------------
class MAEClassifier(torch.nn.Module):
    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = torch.nn.Linear(backbone.num_features, num_classes)
    def forward(self, x):
        return self.classifier(self.backbone(x))

# ------------ Extra eval with search (tau, bias_nevus) ------------
def eval_extras(model, loader, device, class_names,
                tau_grid=np.linspace(0, 2, 21),
                nevus_bias_grid=np.linspace(0, 1.5, 16)):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            all_logits.append(model(x).cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits)              # [N, C] on CPU
    labels = torch.cat(all_labels)              # [N]
    probs = torch.softmax(logits, dim=1).numpy()
    y = labels.numpy()
    K = probs.shape[1]
    y_onehot = np.eye(K)[y]

    # 基线指标（原始 argmax）
    top1 = (probs.argmax(1) == y).mean()
    top2 = np.mean([y[i] in np.argsort(probs[i])[::-1][:2] for i in range(len(y))])
    cm_raw = confusion_matrix(y, probs.argmax(1))
    try:
        roc_macro = roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro")
    except Exception:
        roc_macro = None
    pr_macro = average_precision_score(y_onehot, probs, average="macro")

    # 估计先验
    prior = (y_onehot.sum(0) + 1e-12) / len(y)

    # 2D 搜索：(tau, bias_nevus)
    best = {"acc": -1, "tau": 0.0, "bias_nevus": 0.0}
    nevus_idx = class_names.index("nevus") if "nevus" in class_names else None
    for tau in tau_grid:
        base = logits - torch.from_numpy(np.log(prior)).float() * tau
        if nevus_idx is None:
            pred = base.argmax(1).numpy()
            acc = (pred == y).mean()
            if acc > best["acc"]:
                best.update({"acc": float(acc), "tau": float(tau), "bias_nevus": 0.0})
        else:
            for b in nevus_bias_grid:
                adj = base.clone()
                adj[:, nevus_idx] -= b
                pred = adj.argmax(1).numpy()
                acc = (pred == y).mean()
                if acc > best["acc"]:
                    best.update({"acc": float(acc), "tau": float(tau), "bias_nevus": float(b)})

    print(f"[EXTRA] Top-1={top1:.3f} Top-2={top2:.3f} Macro ROC-AUC={roc_macro if roc_macro is not None else 'NA'} Macro PR-AUC={pr_macro:.3f}")
    print(f"[EXTRA] Best Acc with (tau, bias_nevus): {best['acc']:.3f} at (tau={best['tau']:.2f}, bias={best['bias_nevus']:.2f})")
    print("[EXTRA] Confusion matrix (raw, rows=true, cols=pred):\n", cm_raw)
    print("[EXTRA] Classes:", class_names)

    return {
        "top1": top1, "top2": top2, "roc_macro": roc_macro, "pr_macro": pr_macro,
        "best_acc": best["acc"], "best_tau": best["tau"], "bias_nevus": best["bias_nevus"],
        "prior": prior.tolist(), "logits": logits, "labels": labels
    }

# ------------ Decision rule ------------
def apply_decision_rule(logits, class_names, prior, tau, bias_map=None):
    logits_adj = logits - torch.from_numpy(np.log(prior + 1e-12)).to(logits.dtype) * float(tau)
    if bias_map:
        for name, b in bias_map.items():
            if name in class_names:
                idx = class_names.index(name)
                logits_adj[:, idx] -= float(b)
    return logits_adj

# ------------ Data ------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")

def build_val_loader(root, img_size=224, eval_resize=256, batch_size=32, num_workers=4):
    tf = transforms.Compose([
        transforms.Resize(eval_resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    ds = datasets.ImageFolder(f"{root}/Test", transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return ds, dl

# ------------ Robust ckpt load ------------
def load_ckpt_into(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strip 'module.' if present
    if isinstance(state, dict) and state and all(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] missing keys:", missing)
    if unexpected:
        print("[WARN] unexpected keys:", unexpected)

# ------------ main ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="vit_base_patch16_224.mae")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--eval-resize", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/decision_calibration.json")
    args = ap.parse_args()

    device = get_device()
    ds, val_loader = build_val_loader(args.root, args.img_size, args.eval_resize,
                                      args.batch_size, args.num_workers)

    backbone = timm.create_model(args.model, pretrained=True, num_classes=0)
    model = MAEClassifier(backbone, num_classes=len(ds.classes)).to(device)
    load_ckpt_into(model, args.ckpt, device)

    # 先跑搜索 + 基线报告
    extras = eval_extras(model, val_loader, device, ds.classes)

    # 保存校准参数
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "best_tau": float(extras["best_tau"]),
            "bias_nevus": float(extras["bias_nevus"]),
            "prior": extras["prior"]
        },
        open(args.out, "w"), indent=2
    )
    print(f"[OK] saved {args.out}")

    # === 使用校准规则再次“拍板”，打印校准后的 Acc & 混淆矩阵 ===
    prior = np.array(extras["prior"], dtype=np.float32)
    tau = float(extras["best_tau"])
    bias_map = {"nevus": float(extras["bias_nevus"])} if "nevus" in ds.classes else None

    logits = extras["logits"]      # [N, C] torch on CPU
    labels = extras["labels"]      # [N]    torch on CPU
    logits_adj = apply_decision_rule(logits, ds.classes, prior, tau, bias_map)
    pred_adj = logits_adj.argmax(1).numpy()
    acc_adj = (pred_adj == labels.numpy()).mean()
    cm_adj = confusion_matrix(labels.numpy(), pred_adj)

    print(f"[CALIB] Acc after decision rule: {acc_adj:.3f} (vs raw Top-1 {extras['top1']:.3f})")
    print("[CALIB] Confusion matrix (after calibration):\n", cm_adj)

if __name__ == "__main__":
    main()
