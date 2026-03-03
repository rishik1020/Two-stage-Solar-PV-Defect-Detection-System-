🌞 Automated Solar Panel Defect Detection using ConvNeXt

An intelligent deep learning system for automated surface defect detection in solar photovoltaic (PV) modules using RGB images and transfer learning.

📌 Overview

Solar panel efficiency is significantly affected by surface defects such as cracks, dust accumulation, bird droppings, and electrical damage. Manual inspection is costly and inefficient for large solar farms.

This project proposes a hybrid intelligent inspection framework that integrates:

📡 Sensor-based performance monitoring

📷 Automated image acquisition

🧠 ConvNeXt-based deep learning classification

📊 Ensemble learning with Test-Time Augmentation

The final system achieves 97%+ classification accuracy and is designed to be scalable for real-world solar farm deployment.

🧠 Defect Classes

The model classifies solar panel images into the following 6 categories:

Bird-drop

Clean

Dusty

Electrical-Damage

Physical-Damage

Snow-Covered

🏗 System Architecture
Stage 1 – Sensor-Based Monitoring

Voltage & Current sensors continuously monitor panel performance

Power = Voltage × Current

Abnormal drop → panel flagged for inspection

Stage 2 – Image Acquisition

RGB cameras capture panel images

Images sent to AI classification system

Stage 3 – AI-Based Classification

ConvNeXt-Base (ImageNet pretrained)

Fine-tuned on merged 6-class dataset

Ensemble + TTA for robust inference

🚀 Model Highlights

ConvNeXt-Base backbone

Transfer learning

Progressive image resizing (320 → 448 px)

Mixup & CutMix augmentation

Focal Loss

5-Fold Stratified Cross Validation

Ensemble Learning

Test-Time Augmentation (TTA)

📊 Performance
Model	Accuracy
ResNet50	~91%
EfficientNetB0	~93%
ConvNeXt-Base	96%+
Ensemble + TTA	97%+

ROC-AUC > 0.97 for all classes

Balanced Precision & Recall

Robust under lighting variations

🛠 Tech Stack

Python

PyTorch

timm

Albumentations

Scikit-learn

NumPy

Matplotlib

CUDA (GPU acceleration)
