# 🌾 Transformers for Rice Leaf Disease Classification 🔬

This repository contains the codebase used for fine-tuning, evaluating, and analyzing deep learning models applied to rice leaf disease classification. The project is part of the research paper:

**"Transformers for Rice Leaf Disease Classification: Evaluating Performance–Efficiency Trade-offs and Misclassification Patterns"**  

## 📖 Project Overview

Rice leaf diseases pose a serious threat to food security and farmer livelihoods, especially in the Philippines. This project investigates the application of modern machine vision techniques—specifically transformer-based and convolutional neural network (CNN) architectures—for automated classification of rice leaf diseases from field-captured images.

The study evaluates a diverse range of models, from high-capacity architectures to lightweight networks optimized for deployment in resource-constrained environments. The objective is to strike a balance between classification accuracy and computational efficiency, enabling real-time disease detection on mobile or UAV platforms.

In addition to performance benchmarking, the project analyzes common misclassification patterns, offering insights into model limitations and suggesting strategies for improving classification performance across disease types.

## 📂 Dataset

We use a curated rice leaf disease dataset consisting of 8 classes captured under natural field conditions, comprising ~1,500 images split into training and validation sets.

The dataset includes diseases such as:  
- Bacterial Leaf Blight  
- Brown Spot  
- Leaf Blast  
- Leaf Scald  
- Narrow Brown Spot  
- Rice Hispa  
- Sheath Blight  
- Healthy Rice Leaf  

Standard data augmentations such as flips, rotations, color jittering, and Gaussian blur are applied to improve generalization.

## ⚙️ Models

| Model Type           | Example Models                           | Parameter Count       |
|----------------------|----------------------------------------|----------------------|
| High-Capacity Models | ConvNeXtV2, ViT Hybrid, ViT, Swin, DeiT, DinoV2    | ~80M and ~20M        |
| Lightweight Models   | MobileViT, EfficientViT, EfficientFormer, EfficientNet | ~5M to 12M           |

## 📊 Results Summary
- **ViT Hybrid** achieved the highest accuracy but required substantial computational resources.  
- **Lightweight models** like **EfficientViT** offered competitive accuracy while significantly reducing latency and memory usage — making them suitable for deployment on mobile or drone platforms.  
- **Misclassifications** primarily occurred among visually similar diseases, emphasizing the importance of enhanced datasets and more detailed annotations.  
