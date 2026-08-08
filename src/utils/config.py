"""
Project configuration for the HAM10000 Project.

This module contains all project paths and global settings.
"""

from pathlib import Path

# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

LOG_DIR = PROJECT_ROOT / "logs"

# Dataset paths

IMAGES_PART1 = DATASET_DIR / "HAM10000_images_part_1"

IMAGES_PART2 = DATASET_DIR / "HAM10000_images_part_2"

HAM10000_METADATA = DATASET_DIR / "HAM10000_metadata.csv"

HMNIST_8_8_L = DATASET_DIR / "hmnist_8_8_L.csv"

HMNIST_8_8_RGB = DATASET_DIR / "hmnist_8_8_RGB.csv"

HMNIST_28_28_L = DATASET_DIR / "hmnist_28_28_L.csv"

HMNIST_28_28_RGB = DATASET_DIR / "hmnist_28_28_RGB.csv"

# Output paths

QUALITY_REPORT_DIR = OUTPUT_DIR / "quality_reports"

ENHANCED_IMAGE_DIR = OUTPUT_DIR / "enhanced_images"

VALIDATION_REPORT_DIR = OUTPUT_DIR / "validation_reports"

MODEL_DIR = OUTPUT_DIR / "models"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

PLOTS_DIR = OUTPUT_DIR / "plots"

PREDICTION_DIR = OUTPUT_DIR / "predictions"

# Create required directories

DIRECTORIES = [
    OUTPUT_DIR,
    LOG_DIR,
    QUALITY_REPORT_DIR,
    ENHANCED_IMAGE_DIR,
    VALIDATION_REPORT_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    PLOTS_DIR,
    PREDICTION_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset settings

SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
)

IMAGE_SIZE = (224, 224)

IMAGE_CHANNELS = 3

NUM_CLASSES = 7

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

# Reproducibility

RANDOM_SEED = 42

# Training settings

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 1e-4

VALIDATION_SPLIT = 0.15

TEST_SPLIT = 0.15

# CQI settings

CQI_WEIGHTS = {
    "contrast": 0.20,
    "brightness": 0.20,
    "blur": 0.20,
    "noise": 0.20,
    "hair": 0.20,
}

# Visualization settings

FIGURE_SIZE = (8, 6)

DPI = 300

# File names

QUALITY_REPORT_FILE = QUALITY_REPORT_DIR / "quality_report.csv"

SUMMARY_REPORT_FILE = QUALITY_REPORT_DIR / "summary_statistics.csv"

PQV_REPORT_FILE = VALIDATION_REPORT_DIR / "pqv_report.csv"

BEST_MODEL_PATH = MODEL_DIR / "best_model.keras"

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_checkpoint.keras"

LOG_FILE = LOG_DIR / "project.log"