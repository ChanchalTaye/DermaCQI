# ====================== testing.py ======================
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, roc_auc_score, classification_report, 
                             confusion_matrix)
import pandas as pd
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from dataloader import create_dataloaders
from model import MobileNetV3Classifier

RESULTS_DIR = r"D:\Projects\DERMACQI\models\MobileNetV3_Large"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16, pad=20)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=13)
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_model(model_name="MobileNetV3_Large"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, _, test_loader, class_names, weights, _ = create_dataloaders(batch_size=8, verbose=False)
    
    model = MobileNetV3Classifier(num_classes=7, pretrained=False)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_{model_name}.pth')
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded best model from: {ckpt_path}")
    
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    print("\n" + "="*65)
    print(f"TEST RESULTS - {model_name} (HAM10000 @ 384×384)")
    print("="*65)
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("="*65)
    
    report = classification_report(all_labels, all_preds, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    # Save JSON
    results_dict = {
        "overall": {"accuracy": round(acc, 6), "auc": round(auc, 6),
                    "precision": round(prec, 6), "recall": round(rec, 6), "f1_score": round(f1, 6)},
        "per_class": {}
    }
    for cls in class_names:
        results_dict["per_class"][cls] = {
            "precision": round(report[cls]["precision"], 4),
            "recall": round(report[cls]["recall"], 4),
            "f1_score": round(report[cls]["f1-score"], 4),
            "support": int(report[cls]["support"])
        }
    
    json_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_test_results.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"✅ JSON saved → {json_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, f'Confusion Matrix - {model_name}', cm_path)
    print(f"✅ Confusion Matrix saved → {cm_path}")
    
    # Per-class CSV
    per_class_df = pd.DataFrame.from_dict(results_dict["per_class"], orient='index')
    per_class_df.index.name = 'class'
    csv_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_per_class_metrics.csv")
    per_class_df.to_csv(csv_path)
    print(f"✅ Per-class CSV saved → {csv_path}")

if __name__ == "__main__":
    test_model(model_name="MobileNetV3_Large")