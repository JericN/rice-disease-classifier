# 🧠 Fine-Tuning Pipeline User Guide

> For fine-tuning, this project uses the `HuggingFace Trainer API` for seamless training, evaluation, and logging.

This guide walks you through using the fine-tuning pipelines to train image classification models (ViTs, CNNs, hybrids) using model checkpoints and datasets hosted on Hugging Face.


## 📌 Overview

The fine-tuning pipeline supports training pretrained models with datasets and models stored in [🤗 Hugging Face](https://huggingface.co/). You will need both a **Google account** and a **Hugging Face account**.

### ✅ What It Does
- Fine-tunes a Hugging Face model using a processed and split image dataset from Hugging Face Hub
- Uploads the trained model and model card to your Hugging Face account
- Saves checkpoints and logs to your **Google Drive**
- Generates training metrics: **TensorBoard logs** and **Excel files**


## 📁 Requirements

- ✅ Google Account (for saving to Google Drive)
- ✅ Hugging Face Account (for accessing and uploading models/datasets)


## 📥 Required Inputs

| Parameter               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `hf_model_path`         | Hugging Face model path (e.g. `google/vit-base-patch16-224`)               |
| `hf_dataset_path`       | Hugging Face dataset path (must have `train`, `validation`, and `test` splits) |
| `training_epoch`        | Number of training epochs                                                  |
| `resume_from_checkpoint`| Resume training from existing checkpoint                |
| `output_dir`            | Google Drive directory to save model checkpoints and training results      |

> ⚠️ Ensure the dataset is **preprocessed** and split into `train`, `validation`, and `test`.  
> This pipeline assumes preprocessing and augmentation are already done.


## 🧠 Choosing a Fine-Tuning Strategy

There are two notebooks provided:

| Notebook                      | Strategy |
|------------------------------|----------|
| `train_partial_finetune.ipynb` | 🔧 Fine-tunes **only the classifier head** |
| `train_full_finetune.ipynb`    | 🔁 Fine-tunes the **entire model**, including the backbone |


## ⚙️ Configuring Parameters

Locate the section in the notebook where you define:

- `model_name` – set this to your Hugging Face model checkpoint
- `dataset_name` – the Hugging Face dataset you uploaded
- `output_dir` – Google Drive path to store results
- `resume_from_checkpoint` – set to a path or `None`
- `num_train_epochs` – total number of training epochs

These variables are passed into the training pipeline automatically.


## 🛠️ Setting Additional Training Arguments

You can modify the training behavior by editing the `TrainingArguments` block in the notebook. Parameters supported by the Hugging Face Trainer API include:

- `learning_rate` – customize the optimizer's learning rate
- `per_device_train_batch_size` – control the batch size per GPU/TPU core
- `load_best_model_at_end` – automatically restore the best checkpoint
- `metric_for_best_model` – specify which metric to monitor (e.g. `accuracy`)
- `fp16` – enable mixed precision training on supported hardware

More options can be configured using the Trainer API
> These arguments offer full flexibility to tailor the training process for your specific model and dataset.


## 🏁 Output

After execution, the pipeline will:

- ✅ Upload the **trained model** and **model card** to Hugging Face Hub
- ✅ Save a **checkpoint directory** to your specified Google Drive folder
- ✅ Log training metrics in **TensorBoard**
- ✅ Export training summaries as **Excel spreadsheets**


## 🚀 Workflow Summary

1. Upload your dataset to Hugging Face Hub using `dataset_upload.ipynb`
2. Open either `train_partial_finetune.ipynb` or `train_full_finetune.ipynb`
3. Set the model name, dataset path, output directory, number of epochs, and checkpoint options
4. Modify training arguments in the `TrainingArguments` block if needed
5. Run all cells to start fine-tuning

Monitor your model's progress via:
- Hugging Face Model Hub (for logs, card, and model files)
- Google Drive (for checkpoints and Excel summaries)


## 💬 Questions?

For support or advanced customization, refer to notebook comments or consult the [Hugging Face Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer).

---
