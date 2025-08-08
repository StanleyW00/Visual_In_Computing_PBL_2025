#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import Counter
from torchvision import datasets

def load_split(root: Path, split: str):
    split_dir = root / split
    if not split_dir.exists():
        print(f"[WARN] {split_dir} not found, skip.")
        return None
    ds = datasets.ImageFolder(str(split_dir))
    counts = Counter(ds.targets)  # idx -> count
    per_class = [(cls, counts[i]) for cls, i in ds.class_to_idx.items()]
    per_class.sort(key=lambda x: x[0])
    return {
        "name": split,
        "num_classes": len(ds.classes),
        "classes": ds.classes,
        "total": len(ds.samples),
        "per_class": per_class,
        "class_to_idx": ds.class_to_idx,
    }

def print_stats(info):
    print(f"\n== {info['name']} ==")
    print(f"Num classes: {info['num_classes']}")
    print(f"Total images: {info['total']}")
    # 打印每类计数
    width = max(len(c) for c,_ in info["per_class"]) if info["per_class"] else 0
    for cls, cnt in info["per_class"]:
        print(f"{cls.ljust(width)} : {cnt}")
    # 简单的不平衡度
    if info["per_class"]:
        counts = [cnt for _, cnt in info["per_class"]]
        if min(counts) > 0:
            imb = max(counts) / min(counts)
            print(f"Imbalance ratio (max/min): {imb:.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ISIC_dataset", help="dataset root containing Train/Test")
    ap.add_argument("--splits", nargs="*", default=["Train", "Test"])
    args = ap.parse_args()

    root = Path(args.root)
    infos = []
    for sp in args.splits:
        info = load_split(root, sp)
        if info:
            infos.append(info)
            print_stats(info)

    # 检查不同 split 的类别集合是否一致
    if len(infos) >= 2:
        base = set(infos[0]["classes"])
        for other in infos[1:]:
            s = set(other["classes"])
            only_in_base  = sorted(list(base - s))
            only_in_other = sorted(list(s - base))
            if only_in_base or only_in_other:
                print(f"\n[WARNING] Class mismatch between {infos[0]['name']} and {other['name']}:")
                if only_in_base:
                    print(f"  Only in {infos[0]['name']}: {only_in_base}")
                if only_in_other:
                    print(f"  Only in {other['name']}: {only_in_other}")
            else:
                print(f"\nClasses match between {infos[0]['name']} and {other['name']} ✅")

if __name__ == "__main__":
    main()
