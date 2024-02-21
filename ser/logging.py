from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def log_training(epoch, batch, loss, name):
    with open(PROJECT_ROOT / "experiments" / name / "train_log.txt", "a") as f:
        f.write(f"Train Epoch: {epoch} | Train Loss: {loss:.4f} | Batch: {batch}\n")

def log_validation(epoch, val_loss, val_acc, name):
    with open(PROJECT_ROOT / "experiments" / name / "validation_log.txt", "a") as f:
        f.write(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}\n")