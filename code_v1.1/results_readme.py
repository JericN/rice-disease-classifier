import pandas as pd
from huggingface_hub import hf_hub_download
import re

# Function to load README from a Hugging Face repo
def load_readme(repo_id):
    file_path = hf_hub_download(repo_id=repo_id, filename="README.md")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()

# Function to parse training results from README
def parse_training_results(readme_lines):
    data = []
    capturing = False
    table_pattern = re.compile(r'\|\s*([0-9\.]+)\s*\|\s*([0-9\.]+)\s*\|\s*([0-9\.]+)\s*\|\s*([0-9\.]+)\s*\|\s*([0-9\.]+)\s*\|')
    
    for line in readme_lines:
        if "### Training results" in line:
            capturing = True
            continue
        if capturing and table_pattern.match(line):
            match = table_pattern.match(line)
            if match:
                train_loss, epoch, step, val_loss, accuracy = map(float, match.groups())
                data.append((epoch, accuracy, val_loss, train_loss))
    
    return data

# Function to extract model name from repo_id
def extract_model_name(repo_id):
    return repo_id.split("/")[-1].split("_")[0]  # Extract model name before the first underscore

# Function to compile results into tables
def compile_results(models_path):
    accuracy_data = {}
    val_loss_data = {}
    train_loss_data = {}
    
    for repo_id in models_path:
        model_name = extract_model_name(repo_id)
        readme_lines = load_readme(repo_id)
        training_results = parse_training_results(readme_lines)
        df = pd.DataFrame(training_results, columns=["Epoch", "Accuracy", "Validation Loss", "Training Loss"])
        accuracy_data[model_name] = df.set_index("Epoch")["Accuracy"]
        val_loss_data[model_name] = df.set_index("Epoch")["Validation Loss"]
        train_loss_data[model_name] = df.set_index("Epoch")["Training Loss"]
    
    df_accuracy = pd.DataFrame(accuracy_data)
    df_val_loss = pd.DataFrame(val_loss_data)
    df_train_loss = pd.DataFrame(train_loss_data)
    
    print("Model vs Accuracy per Epoch")
    print(df_accuracy.to_string())
    df_accuracy.to_csv("model_vs_accuracy.csv")
    
    print("\nModel vs Validation Loss per Epoch")
    print(df_val_loss.to_string())
    df_val_loss.to_csv("model_vs_validation_loss.csv")
    
    print("\nModel vs Training Loss per Epoch")
    print(df_train_loss.to_string())
    df_train_loss.to_csv("model_vs_training_loss.csv")

# List of model repositories
models_path = [
    "cvmil/resnet-50_augmented-v2_fft",
    "cvmil/convnext-base-224_augmented-v2_fft",
    "cvmil/vit-base-patch16-224_augmented-v2_fft",
    "cvmil/vit-hybrid-base-bit-384_augmented-v2_fft",
    "cvmil/swin-base-patch4-window7-224_augmented-v2_fft",
    "cvmil/deit-base-patch16-224_augmented-v2_fft",
    "cvmil/beit-base-patch16-224_augmented-v2_fft",
    "cvmil/dinov2-base_augmented-v2_fft",
]

# Compile results for all models
compile_results(models_path)