import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
DATASET_PATH = r"C:\Users\E028.6\Downloads\Merged Fruit Dataset"
OUTPUT_BASE_DIR = r"C:\Users\E028.6\Downloads\Phase-3\Results"
NUM_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

# Model Zoo
MODELS_DICT = {
    "MobileNetV3": "mobilenetv3_large_100",
    "EfficientNetV2": "efficientnet_b0",
    "VisionTransformer": "vit_base_patch16_224",
    "SwinTransformer": "swin_tiny_patch4_window7_224",
    "SE_ResNet50": "seresnet50",          # Attention 1 (Channel Attention)
    "CoAtNet": "coatnet_0_rw_224"         # Attention 2 (Hybrid Self-Attention & Conv)
}

# ==========================================
# 2. TRANSFORMS & DATA PREPARATION
# ==========================================
def get_data_transforms(augment: bool = False):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class CustomSubset(torch.utils.data.Dataset):
    def __init__(self, full_dataset, indices, transform=None):
        self.full_dataset = full_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, target = self.full_dataset.samples[self.indices[idx]]
        img = self.full_dataset.loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img, target

def prepare_datasets(dataset_path: str):
    train_dir = os.path.join(dataset_path, "Train set") if os.path.exists(os.path.join(dataset_path, "Train set")) else dataset_path
    full_raw_dataset = datasets.ImageFolder(train_dir)
    class_names = full_raw_dataset.classes
    num_classes = len(class_names)
    
    total_samples = len(full_raw_dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    train_indices, test_indices = random_split(
        range(total_samples), [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    return full_raw_dataset, train_indices, test_indices, class_names, num_classes

def build_model(model_name: str, num_classes: int):
    timm_identifier = MODELS_DICT[model_name]
    model = timm.create_model(timm_identifier, pretrained=True, num_classes=num_classes)
    return model

# ==========================================
# 3. TRAINING & REPORTING ENGINE
# ==========================================
def run_experiment(model_key: str, mode: str, dataloaders: dict, class_names: list, num_classes: int, exp_dir: str):
    os.makedirs(exp_dir, exist_ok=True)
    prefix = f"{model_key}_{mode}"
    
    # Save Class Mapping
    pd.DataFrame({"Class_Index": range(len(class_names)), "Class_Name": class_names}).to_csv(
        os.path.join(exp_dir, "Class_Mapping.csv"), index=False
    )

    model = build_model(model_key, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Save Model Summary
    with open(os.path.join(exp_dir, "Model_Summary.txt"), "w") as f:
        f.write(str(model))

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n=======================================================")
    print(f" Starting Experiment: {prefix} on {DEVICE}")
    print(f"=======================================================")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects.double() / len(dataloaders[phase].dataset)).item()

            if phase == 'train':
                tr_loss, tr_acc = epoch_loss, epoch_acc
            else:
                te_loss, te_acc = epoch_loss, epoch_acc
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | Train Acc: {tr_acc:.4f} | Train Loss: {tr_loss:.4f} | Test Acc: {te_acc:.4f} | Test Loss: {te_loss:.4f}")

    total_time = time.time() - start_time
    torch.save(best_model_wts, os.path.join(exp_dir, f"{prefix}_Best.pt"))

    # Save History CSV
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(exp_dir, f"{prefix}_Complete_History.csv"), index=False)
    df_hist.to_csv(os.path.join(exp_dir, f"{prefix}_History.csv"), index=False)

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df_hist["epoch"], df_hist["train_acc"], label="Train Accuracy", color="#1f77b4")
    plt.plot(df_hist["epoch"], df_hist["test_acc"], label="Test Accuracy", color="#ff7f0e")
    plt.title(f"{prefix} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, f"{prefix}_Accuracy.png"), bbox_inches='tight')
    plt.close()

    # Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", color="#1f77b4")
    plt.plot(df_hist["epoch"], df_hist["test_loss"], label="Test Loss", color="#ff7f0e")
    plt.title(f"{prefix} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, f"{prefix}_Loss.png"), bbox_inches='tight')
    plt.close()

    # Confusion Matrix & Classification Report on Best Weights
    model.load_state_dict(best_model_wts)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(exp_dir, f"{prefix}_Confusion_Matrix.csv")
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{prefix} Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(os.path.join(exp_dir, f"{prefix}_Confusion_Matrix.png"), bbox_inches='tight')
    plt.close()

    cls_report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(os.path.join(exp_dir, f"{prefix}_Classification_Report.txt"), "w") as f:
        f.write(cls_report)

    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=range(len(class_names)))
    df_per_class = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": support
    })
    df_per_class.to_csv(os.path.join(exp_dir, f"{prefix}_Per_Class_Metrics.csv"), index=False)

    final_train_acc = df_hist["train_acc"].iloc[-1]
    final_test_acc = df_hist["test_acc"].iloc[-1]
    final_train_loss = df_hist["train_loss"].iloc[-1]
    final_test_loss = df_hist["test_loss"].iloc[-1]

    with open(os.path.join(exp_dir, f"{prefix}_Final_Metrics.txt"), "w") as f:
        f.write(f"Model: {model_key}\nMode: {mode}\nBest Test Accuracy: {best_acc:.6f}\n"
                f"Final Train Acc: {final_train_acc:.6f}\nFinal Test Acc: {final_test_acc:.6f}\n"
                f"Final Train Loss: {final_train_loss:.6f}\nFinal Test Loss: {final_test_loss:.6f}\n"
                f"Training Duration: {total_time:.2f}s\n")

    return {
        "Model": model_key,
        "Mode": mode,
        "Train Accuracy": final_train_acc,
        "Test Accuracy": final_test_acc,
        "Train Loss": final_train_loss,
        "Test Loss": final_test_loss,
        "Best Test Accuracy": best_acc
    }

# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================
def main():
    print(f"[1/3] Loading dataset from: {DATASET_PATH}")
    full_dataset, train_idx, test_idx, class_names, num_classes = prepare_datasets(DATASET_PATH)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    dist_df = pd.DataFrame({
        "Class": class_names,
        "Count": [full_dataset.targets.count(i) for i in range(num_classes)]
    })
    dist_df.to_csv(os.path.join(OUTPUT_BASE_DIR, "Dataset_Distribution.csv"), index=False)
    print(f"[2/3] Found {num_classes} classes and {len(full_dataset)} total images.")

    summary_records = []
    exp_counter = 1

    print(f"[3/3] Starting model runs across NDA and DA modes...")
    for model_key in MODELS_DICT.keys():
        for mode, augment in [("NDA", False), ("DA", True)]:
            exp_folder_name = f"{exp_counter:02d}_{model_key}_{mode}"
            exp_dir = os.path.join(OUTPUT_BASE_DIR, exp_folder_name)
            
            train_transform = get_data_transforms(augment=augment)
            test_transform = get_data_transforms(augment=False)

            train_ds = CustomSubset(full_dataset, train_idx, transform=train_transform)
            test_ds = CustomSubset(full_dataset, test_idx, transform=test_transform)

            dataloaders = {
                'train': DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
                'test': DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            }

            metrics = run_experiment(model_key, mode, dataloaders, class_names, num_classes, exp_dir)
            summary_records.append(metrics)
            exp_counter += 1

    # Save consolidated summary across all models
    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(OUTPUT_BASE_DIR, "Overall_Fruit_Ripeness_Master_Summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "="*60)
    print("ALL RUNS COMPLETE!")
    print(f"Consolidated summary saved to: {summary_path}")
    print("="*60)

if __name__ == "__main__":
    main()