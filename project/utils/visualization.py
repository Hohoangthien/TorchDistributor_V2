import os
import json
import tempfile
import shutil
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .hdfs_utils import upload_local_directory_to_hdfs

def plot_and_save_history(
    train_losses, train_accuracies, val_losses, val_accuracies, output_dir, model_type
):
    try:
        local_tmp_dir = tempfile.mkdtemp()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        epochs = range(1, len(train_losses) + 1)
        
        ax1.plot(epochs, train_losses, "b-o", label="Training Loss")
        ax1.plot(epochs, val_losses, "r-o", label="Validation Loss")
        ax1.set_title(f"{model_type.upper()} Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, train_accuracies, "b-o", label="Training Accuracy")
        ax2.plot(epochs, val_accuracies, "r-o", label="Validation Accuracy")
        ax2.set_title(f"{model_type.upper()} Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(local_tmp_dir, f"training_history_{model_type}.png"), dpi=300)
        plt.close()

        history_data = {
            "model_type": model_type,
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
        }
        with open(os.path.join(local_tmp_dir, f"training_history_{model_type}.json"), "w") as f:
            json.dump(history_data, f, indent=2)

        if output_dir.startswith("hdfs://"):
            upload_local_directory_to_hdfs(local_tmp_dir, output_dir)
        shutil.rmtree(local_tmp_dir)
    except Exception as e:
        print(f"[ERROR] Failed to plot history: {e}")

def plot_and_save_confusion_matrix(cm, class_names, output_dir, model_type):
    """Vẽ và lưu ma trận nhầm lẫn dưới dạng heatmap."""
    try:
        local_tmp_dir = tempfile.mkdtemp()
        
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        figsize = (max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6))

        plt.figure(figsize=figsize)
        
        sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='coolwarm', 
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f"{model_type.upper()} Model - Confusion Matrix (%)", fontsize=14)
        plt.xlabel("Predicted Labels", fontsize=12)
        plt.ylabel("True Labels", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        plot_path = os.path.join(local_tmp_dir, f"confusion_matrix_{model_type}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Confusion matrix plot saved locally to {plot_path}")

        if output_dir.startswith("hdfs://"):
            upload_local_directory_to_hdfs(local_tmp_dir, output_dir)
            print(f"[INFO] Successfully uploaded confusion_matrix_{model_type}.png to {output_dir}")
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to plot or save confusion matrix: {e}")
        traceback.print_exc()
    finally:
        shutil.rmtree(local_tmp_dir)
