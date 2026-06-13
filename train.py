import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, device, model_name,
                epochs=100, lr=5e-3, weighted_class=None):
    """Training loop + validation. Return best model, history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)

    print("\nStart Training....")
    log_file = f"training_{model_name}.log"
    best_model = copy.deepcopy(model)
    best_val_acc = 0.0
    loss_hist, acc_hist, val_loss_hist, val_acc_hist = [], [], [], []

    for epoch in range(epochs):
        t_ep = time.time()
        model.train()
        running_loss, correct = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            if model_name == 'yolo':
                outputs = outputs[:]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        accuracy = correct / len(train_loader.dataset)
        loss_hist.append(avg_loss)
        acc_hist.append(accuracy)

        model.eval()
        val_loss, correct_val = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if model_name == 'yolo':
                    outputs = outputs[0]
                val_loss += criterion(outputs, labels).item() * labels.size(0)
                correct_val += (outputs.argmax(1) == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val / len(val_loader.dataset)
        val_loss_hist.append(avg_val_loss)
        val_acc_hist.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), f"best_model_{model_name}.pt")

        ep_time = time.time() - t_ep
        log_line = (f"Epoch [{epoch+1}/{epochs}]  "
                    f"Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}  "
                    f"Val_Loss: {avg_val_loss:.4f}  Val_Acc: {val_acc:.4f}  "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}  "
                    f"Time: {ep_time:.1f}s")
        print(log_line)
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

    history = {
        'loss': loss_hist, 'acc': acc_hist,
        'val_loss': val_loss_hist, 'val_acc': val_acc_hist,
    }
    return best_model, best_val_acc, history

def plot_training_curves(history, model_name):
    """Plot loss & accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss Curve')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy Curve')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"training_curves_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
