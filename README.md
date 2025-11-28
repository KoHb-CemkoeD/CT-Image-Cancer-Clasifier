# CT-Image-Cancer-Clasifier
## Attention-based Deep Learning for Lung CT Cancer Classification

This repository contains the implementation of a project on **attention mechanisms in deep learning** for **automatic classification of lung CT images** into cancer / non-cancer classes. The project compares two convolutional neural network architectures — a baseline CNN and an attention-enhanced model — and provides a **graphical Qt application** for end users using **Windows GUI**.   

---

## Table of Contents
- [Research Summary](#-research-summary)
- [Project Goals](#-project-goals)
- [Features](#-features)
- [Environment](#-environment)
- [Features](#-features)
- [Implementation](#implementation)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Experiments & Results](#experiments--results)

---

## 🔍 Research summary

This project explores how attention mechanisms in deep learning can improve the classification of lung CT images for early cancer detection. We implement and compare two convolutional neural network architectures:

- **Baseline model** based on a pre-trained EfficientNet backbone.
- **Attention model** that augments the backbone with a custom attention block focusing on diagnostically important image regions.

The models predict four classes: **three lung cancer types** (e.g. adenocarcinoma, large-cell carcinoma, squamous cell carcinoma) and **“no cancer”**. The code includes:

- model architectures and training/evaluation scripts (PyTorch),
- image preprocessing pipeline (PIL, NumPy),
- a **Qt-based GUI application** for Windows allowing a user to load a CT image and obtain an automatic prediction (presence and type of cancer).   

Experimental comparison shows that the attention-enhanced network can detect about **90% of cancer cases**, whereas the baseline without attention achieves only **30–40%**, at the cost of roughly **33.24% more computational complexity**. 

---

## 🎯 Project goals

1. **Study and systematize** attention mechanisms in deep learning in the context of image recognition tasks.
2. **Design and implement** a CNN architecture with an attention module for lung CT classification.
3. **Compare** the effectiveness of:
   - a standard CNN (without attention),
   - a CNN with an attention mechanism,
   for the classification of lung cancer CT images.
4. **Develop a Windows GUI application** for automated CT image classification using the trained model(s).   
5. **Analyse economic efficiency** and potential applicability of the software solution in medical practice.

---

## ✨ Features

- End-to-end pipeline for **lung CT image classification**.
- Two model variants:
  - baseline CNN using a pre-trained **EfficientNet** backbone,
  - attention-enhanced CNN with a custom multi-layer attention block.
- Implementation in **Python** with **PyTorch** (and optionally TensorFlow for comparison/experiments).   
- Image preprocessing using **PIL** and **NumPy** (resize, normalization, conversion to tensors).
- Training scripts with configurable hyperparameters.
- Evaluation utilities: accuracy, ROC-AUC, confusion matrices.
- **Qt-based GUI** for Windows:
  - load CT image from file,
  - run classification,
  - show predicted class and confidence, optionally with marked regions.

<img width="375" height="444" alt="image" src="https://github.com/user-attachments/assets/a47f7a00-1a8e-4273-8819-0a11ba35b256" />
    
- Clear separation between:
  - research code (models, experiments),
  - application code (GUI),
  - documentation.

---

## 🧩 Environment

- **Python:** 3.11  
- **Framework:** PyTorch (CUDA-enabled to accelerate tensor operations, convolutional layers, and backpropagation)  
- **Hardware:** GPU-accelerated environment (NVIDIA CUDA by RTX 3060Ti)  
- **Libraries:** NumPy, TensorFlow/Keras, Scikit-learn, Matplotlib
- **Platform:** Windows 10

---

## Implementation

The project is implemented as a modular system with two main parts:  
1) an **image processing module** that loads, preprocesses and classifies CT images using deep neural networks, and  
2) a **graphical desktop application** for Windows that provides an intuitive user interface for clinicians or researchers. 

<img width="454" height="400" alt="image" src="https://github.com/user-attachments/assets/62159037-9117-4b5d-80bc-c0ec92dbbf63" />


The core application logic is organized around three main classes:

- `MainApplication` – entry point of the program, responsible for initializing the image processing module and the main window.
- `MainWindow` – main GUI form that allows the user to:
  - open a CT image from file,
  - start the classification procedure,
  - view and save recognition results.
- `ImageProcessor` – encapsulates the deep learning pipeline:
  - loading the trained neural network model with or without attention,
  - reading and preprocessing the input image,
  - running inference and generating a marked output image with detected cancer regions, if available.   

The overall algorithm includes: loading a CT image, cleaning and normalizing the data, selecting the mathematical model (baseline vs attention), computing predictions, and returning the final class label and visualization to the GUI. 

<img width="474" height="463" alt="image" src="https://github.com/user-attachments/assets/ae8e9ef4-6ff9-4272-a193-364f6cbc464c" />


---

## Model architecture

The project uses a **convolutional neural network with an attention mechanism** built on top of a pre-trained **EfficientNet (EFNS)** backbone with ImageNet weights.   

<img width="475" height="239" alt="image" src="https://github.com/user-attachments/assets/a10457bb-64e6-4c62-b5d6-38e20c6b769f" />

- **Input:** CT images resized to `(224, 224, 3)` RGB.
- **Feature extractor `F`**: EfficientNet processes the image and produces high-level feature maps `f = F(X)`.  
- **Attention block `A`**:
  - several 2D convolutional layers with ReLU activations,
  - final 1-channel convolution with sigmoid activation to generate an attention mask `a = A(f)` that highlights diagnostically important regions.   
- **Masking and pooling**:
  - the attention mask reweights the feature maps,
  - global average pooling aggregates information over spatial dimensions.
- **Classification head `C`**:
  - dropout,
  - fully connected layer(s) with ReLU,
  - final layer with four outputs (three lung cancer types + “no cancer”) and sigmoid/softmax activation for multi-class prediction.   

For comparison, a **baseline network** with the same EfficientNet backbone but **without the attention block** is used as a reference architecture to isolate the effect of attention on classification quality.   

<img width="474" height="227" alt="image" src="https://github.com/user-attachments/assets/ca979113-2851-4b60-92a3-bd7efdcd21af" />

---

## Datasets

<img width="474" height="192" alt="image" src="https://github.com/user-attachments/assets/47c4376a-3938-4f1e-96d2-41d9c94c44e5" />

Training and evaluation are based on a **public Kaggle dataset** of chest CT images containing several types of lung cancer and healthy cases:

- **Source:** Kaggle CT lung cancer dataset.  
- **Classes:**
  - adenocarcinoma,
  - large-cell carcinoma,
  - squamous cell carcinoma,
  - images with **no cancer**.
- **Total size:** approximately **1,000 CT images**.
- **Split:**
  - 70% – training set,
  - 20% – evaluation set (during training),
  - 10% – validation/test set for final assessment of classification performance.

The images are preprocessed (resized, normalized and converted to the required tensor format) before being fed into the network.  

---

## Experiments & results

The experimental study compares two models:

1. **Baseline CNN** without attention.  
2. **Attention-based CNN** with the custom attention module integrated into the EfficientNet backbone.   

Both models are trained on the same training split and evaluated on held-out data. The main performance indicators are **classification accuracy** and detection capability for cancer cases:

- The **attention-based network** is able to detect roughly **90% of lung cancer cases** in the test data.
<img width="263" height="208" alt="image" src="https://github.com/user-attachments/assets/d5faf856-f99d-4749-937e-88e07023eb6b" /> 

- The **baseline network without attention** achieves only about **30–40%** detection.
<img width="255" height="208" alt="image" src="https://github.com/user-attachments/assets/423d6c40-7a64-47c3-ae56-41cfac981016" />

At the same time:

- the number of trainable parameters and computational complexity increase by about **33.24%** when adding the attention mechanism,
- while classification accuracy improves by approximately **50–60%**, which is considered a highly beneficial trade-off for medical image analysis tasks.

The results confirm that focusing the network on the most informative regions of CT scans via attention substantially improves the ability to detect lung cancer, with only a moderate increase in model complexity. 
