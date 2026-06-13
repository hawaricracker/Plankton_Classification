import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report

def evaluate_model(model, test_loader, device, model_name, log_file, size):
    """Evaluasi model pada test set dengan metrik lengkap. Export ONNX."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if model_name == 'yolo':
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    test_acc = (all_preds == all_labels).sum() / len(all_labels)

    cm = confusion_matrix(all_labels, all_preds)

    num_classes = len(np.unique(all_labels))

    recall_per_class = recall_score(all_labels, all_preds, average=None, labels=range(num_classes))
    recall_macro = recall_score(all_labels, all_preds, average='macro', labels=range(num_classes))
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', labels=range(num_classes))

    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=range(num_classes))
    f1_macro = f1_score(all_labels, all_preds, average='macro', labels=range(num_classes))
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', labels=range(num_classes))

    specificity_per_class = []
    for c in range(num_classes):
        tn = cm[:, :].sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        if tn + fp > 0:
            specificity_per_class.append(tn / (tn + fp))
        else:
            specificity_per_class.append(0.0)
    specificity_per_class = np.array(specificity_per_class)
    specificity_macro = np.mean(specificity_per_class)

    try:
        roc_auc_macro = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
        roc_auc_weighted = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
    except Exception as e:
        print(f"ROC-AUC calculation warning: {e}")
        roc_auc_macro, roc_auc_weighted = 0.0, 0.0

    print("\n" + "="*60)
    print(f"TEST EVALUATION METRICS - {model_name}")
    print("="*60)
    print(f"\nAccuracy          : {test_acc:.4f}")
    print(f"Recall (Macro)    : {recall_macro:.4f}")
    print(f"Recall (Weighted) : {recall_weighted:.4f}")
    print(f"F1 Score (Macro)  : {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"Specificity (Macro): {specificity_macro:.4f}")
    print(f"ROC-AUC (Macro)   : {roc_auc_macro:.4f}")
    print(f"ROC-AUC (Weighted): {roc_auc_weighted:.4f}")

    print("\nPer-Class Metrics:")
    print(f"{'Class':>8} | {'Recall':>8} | {'F1':>8} | {'Specificity':>12}")
    print("-" * 45)
    for c in range(num_classes):
        print(f"{c:>8} | {recall_per_class[c]:>8.4f} | {f1_per_class[c]:>8.4f} | {specificity_per_class[c]:>12.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    print("\nConfusion Matrix:")
    print(cm)

    with open(log_file, "a") as f:
        f.write("\n" + "="*60 + "\n")
        f.write("TEST EVALUATION METRICS\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy         : {test_acc:.4f}\n")
        f.write(f"Recall (Macro)   : {recall_macro:.4f}\n")
        f.write(f"F1 Score (Macro) : {f1_macro:.4f}\n")
        f.write(f"Specificity (Macro): {specificity_macro:.4f}\n")
        f.write(f"ROC-AUC (Macro)  : {roc_auc_macro:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(all_labels, all_preds, digits=4))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm) + "\n")

    # ── Plot Confusion Matrix ──────────────────────────────────────────────────
    plt.figure(figsize=(30, 30))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Test Set')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes), rotation=45)
    plt.yticks(tick_marks, range(num_classes))

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved -> confusion_matrix_{model_name}.png")

    # ── Plot ROC Curve ─────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for i, color in zip(range(num_classes), colors):
        y_true_binary = (all_labels == i).astype(int)
        y_score = all_probs[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = roc_auc_score(y_true_binary, y_score)

        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'Class {i} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve - One-vs-Rest (Multiclass)')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"roc_curve_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved -> roc_curve_{model_name}.png")

    # ── Export ONNX ────────────────────────────────────────────────────────────
    dummy = torch.randn(1, 1, size, size).to(device)
    torch.onnx.export(
        model, dummy, f"plankton_classifier_model_{model_name}.onnx",
        input_names=["image"], output_names=["logits"],
        opset_version=18,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Model saved -> plankton_classifier_{model_name}.onnx")

    return {
        'accuracy': test_acc,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'specificity_macro': specificity_macro,
        'roc_auc_macro': roc_auc_macro,
    }
