# ====================== training.py ======================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

# ---------------- DEBUG ----------------
import dataloader
print("=" * 70)
print("Using dataloader from:")
print(dataloader.__file__)
print("=" * 70)

from dataloader import create_dataloaders, set_seed
from model import MobileNetV3Classifier

# ================== RESULTS PATH ==================
RESULTS_DIR = r"D:\Projects\DERMACQI\models\MobileNetV3_Large"

CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = 0.0
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except:
            pass
    return acc, prec, rec, f1, auc

def train_model(model, train_loader, val_loader, weights, device, class_names,
                num_epochs=15, patience=5, model_name="MobileNetV3_Large"):
    
    model = model.to(device)
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
    
    best_val_f1 = 0.0
    no_improve = 0
    history = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        train_preds, train_labels, train_probs = [], [], []
        
        for images, labels in tqdm(train_loader, desc="Train", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # MixUp
            lam = np.random.beta(0.2, 0.2)
            idx = torch.randperm(images.size(0)).to(device)
            mixed = lam * images + (1 - lam) * images[idx]
            labels_a, labels_b = labels, labels[idx]
            
            outputs = model(mixed)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            with torch.no_grad():
                clean_out = model(images)
                preds = torch.argmax(clean_out, dim=1).cpu().numpy()
                probs = F.softmax(clean_out, dim=1).cpu().numpy()
            
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            train_probs.extend(probs)
        
        train_loss = running_loss / len(train_loader)
        train_acc, train_prec, train_rec, train_f1, train_auc = compute_metrics(train_labels, train_preds, np.array(train_probs))
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs)
        
        val_loss /= len(val_loader)
        val_acc, val_prec, val_rec, val_f1, val_auc = compute_metrics(val_labels, val_preds, np.array(val_probs))
        
        print(f" Train: Loss={train_loss:.4f} | Acc={train_acc:.4f} | F1={train_f1:.4f} | AUC={train_auc:.4f}")
        print(f"  Val : Loss={val_loss:.4f} | Acc={val_acc:.4f} | F1={val_f1:.4f} | AUC={val_auc:.4f}")
        
        history.append({
            'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss,
            'train_acc': train_acc, 'val_acc': val_acc,
            'train_f1': train_f1, 'val_f1': val_f1
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({'model_state_dict': model.state_dict()}, 
                       os.path.join(CHECKPOINT_DIR, f'best_{model_name}.pth'))
            print(f" >> Saved best model (Val F1: {val_f1:.4f})")
        
        if val_f1 < best_val_f1 - 1e-4:
            no_improve += 1
        else:
            no_improve = 0
        if no_improve >= patience:
            print("Early stopping triggered.")
            break
        scheduler.step(val_loss)
    
    pd.DataFrame(history).to_csv(os.path.join(CHECKPOINT_DIR, 'train_val_metrics.csv'), index=False)
    print(f"\n✅ Training finished. Results saved in: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, _, class_names, weights, _ = create_dataloaders(batch_size=8, verbose=True)
    
    model = MobileNetV3Classifier(num_classes=7, pretrained=True)
    
    # Freeze early layers of MobileNetV3
    for name, param in model.backbone.named_parameters():
        if "blocks.0" in name or "blocks.1" in name or "blocks.2" in name:
            param.requires_grad = False
    
    train_model(model, train_loader, val_loader, weights, device, class_names, model_name="MobileNetV3_Large")