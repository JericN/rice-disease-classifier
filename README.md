# Transformers for Rice Leaf Disease Classification 

This repository contains the codebase used for fine-tuning, evaluating, and analyzing deep learning models applied to rice leaf disease classification. The project is part of the research paper:

**"Transformers for Rice Leaf Disease Classification: Evaluating Performance–Efficiency Trade-offs and Misclassification Patterns"**  


## Project Overview

Rice leaf diseases threaten food security and farmer livelihoods, particularly in the Philippines. This project explores the use of modern machine vision techniques with transformer-based and convolutional neural network (CNN) architectures to automatically classify rice leaf diseases from images.

The research evaluates a spectrum of models ranging from high-capacity architectures (e.g., ViT Hybrid, ConvNeXt, Swin) to lightweight networks optimized for resource-constrained deployment (e.g., MobileViT, EfficientViT, EfficientFormer). The goal is to balance classification accuracy with computational efficiency, suitable for real-time agricultural applications on edge devices.


## Dataset

We use a curated rice leaf disease dataset consisting of 8 classes captured under natural field conditions, comprising ~1,500 images split into training, validation.

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


## Models

| Model Type           | Example Models                           | Parameter Count       |
|----------------------|----------------------------------------|----------------------|
| High-Capacity Models | ConvNeXtV2, ViT Hybrid, ViT, Swin, DeiT, DinoV2    | ~80M and ~20M        |
| Lightweight Models   | MobileViT, EfficientViT, EfficientFormer, EfficientNet | ~5M to 12M           |



## Results Summary

- **ViT Hybrid** achieved the highest accuracy but required substantial computational resources.
- **Lightweight models** like **EfficientViT** offered competitive accuracy while significantly reducing latency and memory usage — making them suitable for deployment on mobile or drone platforms.
- **Misclassifications** primarily occurred among visually similar diseases, emphasizing the importance of enhanced datasets and more detailed annotations.
