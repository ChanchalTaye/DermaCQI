# ====================== dataloader.py ======================

import os
import random
import warnings

import albumentations as A
import matplotlib
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ============================================================
# PROJECT PATHS
# ============================================================

CSV_PATH = r"D:\Projects\DERMACQI\datasets\HAM10000_metadata.csv"

IMAGE_DIR = r"D:\Projects\DERMACQI\outputs\Enhanced_HAM10000"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class SkinCancerDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, row["image_id"])

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        label = int(row["label"])

        return image, label


# ============================================================
# AUGMENTATION
# ============================================================

def get_transforms():

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=20,
            border_mode=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.30),
        A.Resize(384, 384),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

    return train_transform, test_transform


# ============================================================
# DATALOADER
# ============================================================

def create_dataloaders(batch_size=8, seed=42, verbose=True):

    set_seed(seed)

    # --------------------------------------------------------
    # Read Metadata
    # --------------------------------------------------------

    df = pd.read_csv(CSV_PATH)

    # Add extension if needed
    df["image_id"] = df["image_id"].astype(str)

    df["image_id"] = df["image_id"].apply(
        lambda x: x if x.endswith(".jpg") else x + ".jpg"
    )

    # --------------------------------------------------------
    # Remove Missing Images
    # --------------------------------------------------------

    df["exists"] = df["image_id"].apply(
        lambda x: os.path.exists(os.path.join(IMAGE_DIR, x))
    )

    missing = len(df[df["exists"] == False])

    if missing > 0:
        print(f"Removed {missing} missing images.")

    df = df[df["exists"]].copy()

    # --------------------------------------------------------
    # Encode Labels
    # --------------------------------------------------------

    class_names = [
        "akiec",
        "bcc",
        "bkl",
        "df",
        "mel",
        "nv",
        "vasc"
    ]

    encoder = LabelEncoder()
    encoder.fit(class_names)

    # Encode disease labels from HAM10000 metadata (dx column)
    df["label"] = encoder.transform(df["dx"])

    # --------------------------------------------------------
    # Train / Validation / Test
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=seed
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=seed
    )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    train_transform, test_transform = get_transforms()

    train_dataset = SkinCancerDataset(
        train_df,
        IMAGE_DIR,
        train_transform
    )

    val_dataset = SkinCancerDataset(
        val_df,
        IMAGE_DIR,
        test_transform
    )

    test_dataset = SkinCancerDataset(
        test_df,
        IMAGE_DIR,
        test_transform
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Class Weights
    # --------------------------------------------------------

    class_counts = train_df["label"].value_counts().sort_index()

    weights = torch.tensor(
        np.sqrt(len(train_df) / class_counts.values),
        dtype=torch.float32
    )

    if verbose:

        print("\n" + "=" * 65)
        print("HAM10000 DATASET")
        print("=" * 65)

        print(f"Total Images : {len(df)}")
        print(f"Train Images : {len(train_df)}")
        print(f"Val Images   : {len(val_df)}")
        print(f"Test Images  : {len(test_df)}")

        print("\nClass Mapping")

        for i, c in enumerate(class_names):
            print(f"{i} -> {c}")

        print("\nClass Weights")

        for c, w in zip(class_names, weights):
            print(f"{c:6s}: {w:.4f}")

        print("=" * 65)

    return (
        train_loader,
        val_loader,
        test_loader,
        class_names,
        weights,
        encoder
    )