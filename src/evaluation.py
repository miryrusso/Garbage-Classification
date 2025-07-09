# Metriche
from collections import Counter
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

def plot_class_distribution(loader, class_dict, title='Distribuzione per classi'):
    """
    Plotta la distribuzione delle classi in un dataset PyTorch.

    Args:
        dataset (Dataset): Dataset in cui ciascun elemento è una tupla (immagine, label).
        class_dict (dict): Dizionario che mappa le classi (es: {"cardboard": 1, "glass": 2, ...}).
        title (str): Titolo del grafico.
    """
    labels = [label for _, label in loader.dataset]
    
    label_counts = Counter(labels)

    inv_class_dict = {v - 1: k for k, v in class_dict.items()}

    plt.figure(figsize=(8, 4))
    plt.bar(
        [inv_class_dict[i] for i in sorted(label_counts.keys())],
        [label_counts[i] for i in sorted(label_counts.keys())],
        color='skyblue'
    )
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def evaluate(model, loader, device=None, target_names=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).argmax(1)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Calcolo metriche
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print("\n Confusion Matrix:")
    print(cm)

    if target_names:
        print("\n Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=target_names))

    return acc, prec, rec, f1, cm


def plot_confusion_matrix(model, loader, class_names, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).argmax(1)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_metrics_bar(acc, prec, rec, f1):
    import matplotlib.pyplot as plt

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [acc, prec, rec, f1]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FFC107', '#F44336'])
    plt.ylim(0, 1)
    plt.title('Altre metriche di classificazione')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2 - 0.1, yval + 0.02, f"{yval:.2f}", fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_loss_acc_per_epoch(loss_history_train, loss_history_val, acc_history_train, acc_history_val):
    plt.plot(loss_history_train, label='Train'); plt.plot(loss_history_val, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Cross-entropy loss'); plt.legend(); plt.grid()
    plt.title('Loss curve'); plt.show()

    plt.plot(acc_history_train, label='Train'); plt.plot(acc_history_val, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid()
    plt.title('Accuracy curve'); plt.show()