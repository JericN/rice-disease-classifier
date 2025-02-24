import os
import json
import pandas as pd

def collect_results(json_folder, output_file):
    precision_data = []
    recall_data = []
    f1_data = []
    overall_data = []
    
    for model_dir in os.listdir(json_folder):
        model_path = os.path.join(json_folder, model_dir)
        json_file_path = os.path.join(model_path, "report.json")
        if os.path.isdir(model_path) and os.path.isfile(json_file_path):
            with open(json_file_path, "r") as f:
                data = json.load(f)
                model_name = model_dir
                
                for class_name, metrics in data.items():
                    if isinstance(metrics, dict):
                        precision_data.append({"Model": model_name, "Class": class_name, "Precision": metrics.get("precision", None)})
                        recall_data.append({"Model": model_name, "Class": class_name, "Recall": metrics.get("recall", None)})
                        f1_data.append({"Model": model_name, "Class": class_name, "F1-Score": metrics.get("f1-score", None)})
                
                # Add overall data
                weighted_ave = data["weighted avg"]
                overall_data.append({"Model": model_name, "Accuracy": data.get("accuracy", None), "Precision": weighted_ave.get("precision", None), "Recall": weighted_ave.get("recall", None), "F1-Score": weighted_ave.get("f1-score", None), "Support": weighted_ave.get("support", None), "Eval Time": data.get("evaluation_time_sec", None)})
                    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        pd.DataFrame(precision_data).pivot(index="Model", columns="Class", values="Precision").to_excel(writer, sheet_name="Precision")
        pd.DataFrame(recall_data).pivot(index="Model", columns="Class", values="Recall").to_excel(writer, sheet_name="Recall")
        pd.DataFrame(f1_data).pivot(index="Model", columns="Class", values="F1-Score").to_excel(writer, sheet_name="F1-Score")
        pd.DataFrame(overall_data).to_excel(writer, sheet_name="Overall Accuracy", index=False)
    
    print(f"Results saved to {output_file}")


json_folder = "./results/full fine-tune"
output_file = "./results/full fine-tune.xlsx"
collect_results(json_folder, output_file)