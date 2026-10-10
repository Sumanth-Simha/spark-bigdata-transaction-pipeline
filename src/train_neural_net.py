import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.data_preprocessing import get_dataloaders
from src.models.neural_net import FraudMLP

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def train_model(data_path="data/parquet/creditcard", epochs=10, batch_size=2048, lr=1e-3, device="cpu"):
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, num_features_count = get_dataloaders(data_path, batch_size=batch_size)

    # Define Model
    # Hour has 24 classes, embed dim 4
    embedding_dims = [(24, 4)]
    model = FraudMLP(num_features_count, embedding_dims).to(device)

    # Loss and Optimizer
    # Using Focal Loss for class imbalance
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_pr_auc = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_pr_auc': [], 'val_roc_auc': []
    }

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for x_num, x_cat, labels in train_loader:
            x_num, x_cat, labels = x_num.to(device), x_cat.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(x_num, x_cat)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_num.size(0)

        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for x_num, x_cat, labels in val_loader:
                x_num, x_cat, labels = x_num.to(device), x_cat.to(device), labels.to(device).unsqueeze(1)

                outputs = model(x_num, x_cat)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * x_num.size(0)

                probs = torch.sigmoid(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # Metrics
        pr_auc = average_precision_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_preds)
        history['val_pr_auc'].append(pr_auc)
        history['val_roc_auc'].append(roc_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

        # Save Best Model by PR-AUC
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            os.makedirs("src/models/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "src/models/checkpoints/best_model.pth")
            print("  --> Saved Best Model")

    # Final Evaluation & Export
    model.load_state_dict(torch.load("src/models/checkpoints/best_model.pth"))
    export_reports(model, val_loader, history, device)

def export_reports(model, val_loader, history, device):
    os.makedirs("reports", exist_ok=True)
    model.eval()

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for x_num, x_cat, labels in val_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            outputs = model(x_num, x_cat)
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.numpy())
            all_preds.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Loss Curve
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Focal Loss')
    plt.legend()
    plt.savefig("reports/loss_curve.png")

    # PR Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig("reports/pr_curve.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("reports/roc_curve.png")

    # Confusion Matrix (Threshold 0.5)
    preds_binary = (all_preds > 0.5).astype(int)
    cm = confusion_matrix(all_labels, preds_binary)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Threshold 0.5)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("reports/confusion_matrix.png")

    print("\nFinal Evaluation Metrics (Threshold 0.5):")
    print(f"Precision: {precision_score(all_labels, preds_binary):.4f}")
    print(f"Recall: {recall_score(all_labels, preds_binary):.4f}")
    print(f"F1 Score: {f1_score(all_labels, preds_binary):.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(epochs=5, device=device)
