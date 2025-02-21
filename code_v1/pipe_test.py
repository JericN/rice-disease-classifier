import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoProcessor
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import drive


def evaluate_model(model_name, dataset, labels, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Loads a model and evaluates it on the dataset."""
    print(f"Evaluating {model_name}...")

    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    y_true, y_pred = [], []

    for example in tqdm(dataset, desc=f"Testing {model_name}"):
        image, label = example["image"], example["label"]
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred_label = torch.argmax(outputs.logits, dim=-1).cpu().item()

        y_true.append(label)
        y_pred.append(pred_label)

    return y_true, y_pred


def generate_report(y_true, y_pred, labels, model_name, output_dir):
    """Generates and saves classification report and confusion matrix."""
    model_safe_name = model_name.split("/")[-1]
    model_safe_name = model_safe_name.split("_")[0] + "-tl"
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # Save classification report as JSON
    report_path = os.path.join(output_dir, f"{model_safe_name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    # Save classification report as Excel
    report_df = pd.DataFrame(report).transpose()
    excel_path = os.path.join(output_dir, f"{model_safe_name}_report.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        report_df.to_excel(writer, sheet_name="Classification Report")

        # Save confusion matrix in another sheet
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_excel(writer, sheet_name="Confusion Matrix")

        cm_norm_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
        cm_norm_df.to_excel(writer, sheet_name="Normalized Confusion Matrix")

    # Save confusion matrix plot
    def save_cm_plot(matrix, title, filename, fmt="d"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title, pad=20)
        plt.xticks(rotation=30)
        plt.yticks(rotation=30)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", pad_inches=0.5)
        plt.close()

    save_cm_plot(cm, model_safe_name, f"{model_safe_name}_confusion_matrix.png")
    save_cm_plot(cm_normalized, "Normalized Confusion Matrix", f"{model_safe_name}_normalized_confusion_matrix.png", fmt=".2f")


def main():
    """Mounts Google Drive, loads dataset, evaluates models, and saves reports."""
    drive.mount("/content/drive")

    models_path = [
        "cvmil/resnet-50_rice-leaf-disease-augmented_tl",
        "cvmil/vit-base-patch16-224_rice-leaf-disease-augmented_tl",
        "cvmil/swin-base-patch4-window7-224_rice-leaf-disease-augmented_tl",
        "cvmil/deit-base-patch16-224_rice-leaf-disease-augmented_tl",
        "cvmil/beit-base-patch16-224_rice-leaf-disease-augmented_tl",
        "cvmil/dinov2-base_rice-leaf-disease-augmented_tl",
    ]

    dataset = load_dataset("cvmil/rice-leaf-disease-augmented", split="test")
    labels = dataset.features["label"].names

    output_dir = "/content/drive/Shareddrives/CS198-Drones/test_tl_output/"
    os.makedirs(output_dir, exist_ok=True)

    for model_name in models_path:
        try:
            y_true, y_pred = evaluate_model(model_name, dataset, labels)
            generate_report(y_true, y_pred, labels, model_name, output_dir)
        except Exception as e:
            print(f"⚠️ Error processing {model_name}: {e}")

    print("✅ Evaluation completed. Reports saved to Google Drive.")