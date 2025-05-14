from fastai.vision.all import *
from pathlib import Path
import torch
import os
import timm

os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Set base folder relative to script location
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "high_speed_trains"
MODEL_PATH = BASE_DIR / "train_classifier3.pkl"


def clean_broken_images(path):
    """
    Remove any corrupt or unreadable image files.
    """
    print("🧹 Checking for broken images...")
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(f"✅ Removed {len(failed)} broken images.")


def train_model():
    """
    Train a Fastai model to classify high-speed trains by country.
    """
    print("🚄 Starting training...")
    print(f"📁 Training data folder: {DATA_DIR}")
    print(f"🧠 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"⚡ GPU detected")
    else:
        print("⚠️ GPU not available. Training will use CPU.")

    # Create DataLoaders
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(224),
    ).dataloaders(DATA_DIR, bs=32)

    # Train the model
    learn = vision_learner(dls, "çonvnext_tiny_in22k", metrics=accuracy)
    learn.fine_tune(20, base_lr=1e-3)

    # Save model
    learn.export(MODEL_PATH)
    print(f"\n✅ Model exported to: {MODEL_PATH}")


if __name__ == "__main__":
    # clean_broken_images(DATA_DIR)
    train_model()
