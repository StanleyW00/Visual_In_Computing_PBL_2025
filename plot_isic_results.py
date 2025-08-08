#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot ISIC training/eval results:
- Parses logs from train_isic_mae.py and eval_isic.py
- Saves PNGs: loss/acc/AUC curves, confusion matrices, raw vs calibrated acc

Usage:
  uv run plot_isic_results.py --train_log train.log --eval_log eval.log --outdir plots
"""

import argparse
import os
import re
import ast
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_train_log(path):
    """
    Extract per-epoch metrics from train log.

    Expected lines (examples):
      [Probe][3/8] train loss=1.9492 acc=0.3711 | val loss=2.0439 acc=0.3051 auc=0.7740598654920905
      [FT][10/25] train loss=0.3130 acc=0.8808 | val loss=1.2794 acc=0.6780 auc=0.9386497110921663
    """
    probe_pat = re.compile(
        r"\[Probe\]\[(\d+)/(\d+)\]\s+train loss=([\d.]+)\s+acc=([\d.]+)\s+\|\s+val loss=([\d.]+)\s+acc=([\d.]+)\s+auc=([0-9.NA]+)"
    )
    ft_pat = re.compile(
        r"\[FT\]\[(\d+)/(\d+)\]\s+train loss=([\d.]+)\s+acc=([\d.]+)\s+\|\s+val loss=([\d.]+)\s+acc=([\d.]+)\s+auc=([0-9.NA]+)"
    )

    probe = []
    ft = []
    total_probe = total_ft = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m1 = probe_pat.search(line)
            if m1:
                i, n = int(m1.group(1)), int(m1.group(2))
                total_probe = max(total_probe, n)
                tr_loss, tr_acc = float(m1.group(3)), float(m1.group(4))
                va_loss, va_acc = float(m1.group(5)), float(m1.group(6))
                auc_str = m1.group(7)
                va_auc = float(auc_str) if auc_str != "NA" else np.nan
                probe.append((i, tr_loss, tr_acc, va_loss, va_acc, va_auc))
                continue

            m2 = ft_pat.search(line)
            if m2:
                i, n = int(m2.group(1)), int(m2.group(2))
                total_ft = max(total_ft, n)
                tr_loss, tr_acc = float(m2.group(3)), float(m2.group(4))
                va_loss, va_acc = float(m2.group(5)), float(m2.group(6))
                auc_str = m2.group(7)
                va_auc = float(auc_str) if auc_str != "NA" else np.nan
                ft.append((i, tr_loss, tr_acc, va_loss, va_acc, va_auc))

    # Sort and build continuous epoch index: probe epochs first, then FT with offset
    probe.sort(key=lambda x: x[0])
    ft.sort(key=lambda x: x[0])

    epochs = []
    tr_loss, tr_acc, va_loss, va_acc, va_auc = [], [], [], [], []

    for (i, a, b, c, d, e) in probe:
        epochs.append(i)
        tr_loss.append(a); tr_acc.append(b); va_loss.append(c); va_acc.append(d); va_auc.append(e)

    offset = len(probe)
    for (i, a, b, c, d, e) in ft:
        epochs.append(offset + i)
        tr_loss.append(a); tr_acc.append(b); va_loss.append(c); va_acc.append(d); va_auc.append(e)

    meta = {"probe_epochs": total_probe, "ft_epochs": total_ft}
    return {
        "epochs": np.array(epochs, dtype=int),
        "train_loss": np.array(tr_loss, dtype=float),
        "train_acc": np.array(tr_acc, dtype=float),
        "val_loss": np.array(va_loss, dtype=float),
        "val_acc": np.array(va_acc, dtype=float),
        "val_auc": np.array(va_auc, dtype=float),
        "meta": meta,
    }


def _parse_matrix_block(lines, start_idx, n_rows=None):
    """
    Parse a printed numpy-like matrix starting at start_idx.
    Reads until it has n_rows (if given) or until it sees a line ending with ']'
    and the total bracket balance reaches zero.
    """
    rows = []
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("["):
            break
        # Extract numbers in the current row
        # e.g. "[ 0  1  2]" -> [0,1,2]
        nums = [int(tok) for tok in line.strip("[]").split() if tok.replace("-", "").isdigit()]
        if nums:
            rows.append(nums)
        i += 1
        if n_rows is not None and len(rows) >= n_rows:
            break
    return np.array(rows, dtype=int), i


def parse_eval_log(path):
    """
    Extract summary metrics, classes, and confusion matrices from eval log.

    Expected snippets:
      [EXTRA] Top-1=0.695 Top-2=0.881 Macro ROC-AUC=0.947... Macro PR-AUC=0.820
      [EXTRA] Best Acc with (tau, bias_nevus): 0.737 at (tau=0.40, bias=1.20)
      [EXTRA] Confusion matrix (raw, rows=true, cols=pred):
      [...]
      [EXTRA] Classes: ['a', 'b', ...]
      [CALIB] Acc after decision rule: 0.737 (vs raw Top-1 0.695)
      [CALIB] Confusion matrix (after calibration):
      [...]
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    txt = "".join(lines)

    # Raw top metrics
    m = re.search(r"Top-1=([0-9.]+)\s+Top-2=([0-9.]+)\s+Macro ROC-AUC=([0-9.]+|NA)\s+Macro PR-AUC=([0-9.]+|NA)", txt)
    top1_raw = float(m.group(1)) if m else np.nan
    top2_raw = float(m.group(2)) if m else np.nan
    auc_raw = float(m.group(3)) if (m and m.group(3) != "NA") else np.nan
    pr_auc_raw = float(m.group(4)) if (m and m.group(4) != "NA") else np.nan

    # Best tau/bias
    m2 = re.search(r"Best Acc with \(tau, bias_nevus\):\s*([0-9.]+)\s*at\s*\(tau=([0-9.]+),\s*bias=([0-9.]+)\)", txt)
    best_acc_calib = float(m2.group(1)) if m2 else np.nan
    best_tau = float(m2.group(2)) if m2 else np.nan
    bias_nevus = float(m2.group(3)) if m2 else 0.0

    # Classes
    m3 = re.search(r"\[EXTRA\]\s*Classes:\s*(\[.*\])", txt)
    classes = []
    if m3:
        try:
            classes = ast.literal_eval(m3.group(1))
        except Exception:
            classes = []

    # Calibrated Acc
    m4 = re.search(r"\[CALIB\]\s*Acc after decision rule:\s*([0-9.]+)", txt)
    acc_after = float(m4.group(1)) if m4 else best_acc_calib

    # Parse confusion matrices from line-wise scan (raw then calibrated)
    raw_idx = next((i for i, L in enumerate(lines) if "Confusion matrix (raw" in L), None)
    cal_idx = next((i for i, L in enumerate(lines) if "Confusion matrix (after calibration)" in L), None)

    cm_raw = None
    cm_cal = None
    n_rows = len(classes) if classes else None

    if raw_idx is not None:
        cm_raw, _ = _parse_matrix_block(lines, raw_idx + 1, n_rows=n_rows)
    if cal_idx is not None:
        cm_cal, _ = _parse_matrix_block(lines, cal_idx + 1, n_rows=n_rows)

    return {
        "classes": classes,
        "top1_raw": top1_raw,
        "top2_raw": top2_raw,
        "auc_raw": auc_raw,
        "pr_auc_raw": pr_auc_raw,
        "best_tau": best_tau,
        "bias_nevus": bias_nevus,
        "acc_calibrated": acc_after,
        "cm_raw": cm_raw,
        "cm_cal": cm_cal,
    }


def ensure_outdir(outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)


def plot_curve(x, y_list, labels, title, xlabel, ylabel, outpath):
    plt.figure()
    for y, lab in zip(y_list, labels):
        plt.plot(x, y, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_bar(categories, values, title, ylabel, outpath):
    plt.figure()
    x = np.arange(len(categories))
    plt.bar(x, values)
    plt.xticks(x, categories)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_confusion(cm, classes, title, outpath):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # tick labels
    if classes:
        ticks = np.arange(len(classes))
        plt.xticks(ticks, classes, rotation=45, ha="right")
        plt.yticks(ticks, classes)
    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

import numpy as np

def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_log", required=True, help="Path to train log (stdout from train_isic_mae.py)")
    ap.add_argument("--eval_log", required=True, help="Path to eval log (stdout from eval_isic.py)")
    ap.add_argument("--outdir", default="plots", help="Output directory for figures")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    # ---- Parse logs
    train = parse_train_log(args.train_log)
    evalres = parse_eval_log(args.eval_log)

    # Save parsed metrics for reproducibility
    with open(os.path.join(args.outdir, "parsed_train.json"), "w") as f:
        json.dump(to_jsonable({
            "epochs": train["epochs"],
            "train_loss": train["train_loss"],
            "train_acc": train["train_acc"],
            "val_loss": train["val_loss"],
            "val_acc": train["val_acc"],
            "val_auc": train["val_auc"],
            "meta": train["meta"],
        }), f, indent=2)

    with open(os.path.join(args.outdir, "parsed_eval.json"), "w") as f:
        json.dump(to_jsonable(evalres), f, indent=2)

    # ---- Curves
    plot_curve(
        train["epochs"],
        [train["train_loss"], train["val_loss"]],
        ["Train Loss", "Val Loss"],
        "Loss vs Epoch",
        "Epoch",
        "Loss",
        os.path.join(args.outdir, "loss_vs_epoch.png"),
    )

    plot_curve(
        train["epochs"],
        [train["train_acc"], train["val_acc"]],
        ["Train Acc", "Val Acc"],
        "Accuracy vs Epoch",
        "Epoch",
        "Accuracy",
        os.path.join(args.outdir, "acc_vs_epoch.png"),
    )

    plot_curve(
        train["epochs"],
        [train["val_auc"]],
        ["Val AUC (macro, OvR)"],
        "Validation AUC vs Epoch",
        "Epoch",
        "AUC",
        os.path.join(args.outdir, "auc_vs_epoch.png"),
    )

    # ---- Raw vs Calibrated Acc
    raw_acc = float(evalres["top1_raw"]) if evalres["top1_raw"] == evalres["top1_raw"] else 0.0
    cal_acc = float(evalres["acc_calibrated"]) if evalres["acc_calibrated"] == evalres["acc_calibrated"] else 0.0
    plot_bar(
        ["Raw Top-1", "Calibrated"],
        [raw_acc, cal_acc],
        f"Accuracy: Raw vs Calibrated (tau={evalres['best_tau']:.2f}, bias_nevus={evalres['bias_nevus']:.2f})",
        "Accuracy",
        os.path.join(args.outdir, "raw_vs_calibrated_acc.png"),
    )

    # ---- Confusion matrices
    classes = evalres["classes"]
    if isinstance(classes, list) and all(isinstance(c, str) for c in classes):
        pass
    else:
        classes = [str(i) for i in range(evalres["cm_raw"].shape[0])] if evalres["cm_raw"] is not None else []

    if evalres["cm_raw"] is not None:
        plot_confusion(
            evalres["cm_raw"], classes, "Confusion Matrix (Raw)", os.path.join(args.outdir, "cm_raw.png")
        )
    if evalres["cm_cal"] is not None:
        plot_confusion(
            evalres["cm_cal"], classes, "Confusion Matrix (Calibrated)", os.path.join(args.outdir, "cm_calibrated.png")
        )

    print("[OK] Figures saved to:", args.outdir)
    print(" - loss_vs_epoch.png")
    print(" - acc_vs_epoch.png")
    print(" - auc_vs_epoch.png")
    print(" - raw_vs_calibrated_acc.png")
    if evalres["cm_raw"] is not None:
        print(" - cm_raw.png")
    if evalres["cm_cal"] is not None:
        print(" - cm_calibrated.png")


if __name__ == "__main__":
    main()
