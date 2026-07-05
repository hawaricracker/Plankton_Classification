import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--epoch", type=int, required=True)
parser.add_argument("--size", type=int, required=True)
args = parser.parse_args()

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

import preprocessing
from preprocessing import load_and_filter_data, augment_data, create_dataloaders, set_seed
from model import create_model
from train import train_model, plot_training_curves
from test import evaluate_model

# ── Konfigurasi ───────────────────────────────────────────────────────────────────
preprocessing.SIZE = args.size
SIZE = args.size
maks_sampel = 55000
LR = 5e-3
EPOCHS = args.epoch
BATCH_SIZE = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device       : {device}')

set_seed(13)

if __name__ == '__main__':
    from datasets import load_dataset
    ds = load_dataset("nf-whoi/whoi-plankton-small")

    # ── Load & filter data ──────────────────────────────────────────────────────
    data_train, data_test, data_val, label_counts, new_map_label = load_and_filter_data(ds, maks_sampel)

    # Plot distribusi train sebelum augmentasi
    label_counts_plot = Counter([d['label'] for d in data_train])
    plt.figure()
    plt.bar(list(label_counts_plot.keys()), list(label_counts_plot.values()),
            color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Label'); plt.ylabel('Frequency')
    plt.title('Distribution of Train Image Labels (after filter)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"distribution_train_{args.model}.png", dpi=300, bbox_inches="tight")

    # Plot distribusi test
    plt.figure(figsize=(10, 10))
    label_counts_test = Counter([d['label'] for d in data_test])
    labels = list(label_counts_test.keys())
    counts = list(label_counts_test.values())
    plt.bar(labels, counts, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Label Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Test Image Labels')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"distribution_test_{args.model}.png", dpi=300, bbox_inches="tight")

    # Plot distribusi validation
    plt.figure(figsize=(10, 10))
    label_counts_val = Counter([d['label'] for d in data_val])
    labels = list(label_counts_val.keys())
    counts = list(label_counts_val.values())
    plt.bar(labels, counts, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Label Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Val Image Labels')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"distribution_val_{args.model}.png", dpi=300, bbox_inches="tight")

    label_counts = Counter([d['label'] for d in data_train])
    print(f"Data train setelah filter : {len(data_train)} gambar, "
          f"{len(label_counts)} kelas")

    # Plot distribusi sebelum aug
    plt.figure()
    plt.bar(list(label_counts.keys()), list(label_counts.values()),
            color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Label'); plt.ylabel('Frequency')
    plt.title('Distribution of Train Image Labels (before aug)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"distribution1_{args.model}.png", dpi=300, bbox_inches="tight")
    plt.close()

    weighted_class = [
        len(data_train) / (len(label_counts) * label_counts[i]) for i in label_counts
    ]

    # ── Augmentasi sequential ──────────────────────────────────────────────────
    train_img = augment_data(data_train, label_counts)

    # Plot distribusi setelah augmentasi
    label_counts1 = Counter([d['label'] for d in train_img])
    plt.figure()
    plt.bar(list(label_counts1.keys()), list(label_counts1.values()),
            color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Label'); plt.ylabel('Frequency')
    plt.title('Distribution of Train Image Labels (after aug)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"distribution2_{args.model}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Label unik : {np.unique([d['label'] for d in train_img])}")

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        train_img, data_val, data_test, BATCH_SIZE
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    num_classes = len(label_counts)
    model = create_model(args.model, num_classes, device, input_size=SIZE)

    # ── Training ───────────────────────────────────────────────────────────────
    best_model, best_val_acc, history = train_model(
        model, train_loader, val_loader, device, args.model,
        epochs=EPOCHS, lr=LR, weighted_class=weighted_class
    )

    # ── Plot training curves ───────────────────────────────────────────────────
    plot_training_curves(history, args.model)

    # ── Test evaluasi ──────────────────────────────────────────────────────────
    log_file = f"training_{args.model}.log"
    metrics = evaluate_model(best_model, test_loader, device, args.model, log_file, SIZE)
