# 🧠 Deep Learning Assignments - CSE4261

Welcome to the **Deep Learning AI Repository**! This repo contains 8 hands-on assignments exploring key deep learning topics such as CNNs, activation functions, transfer learning, adversarial attacks, feature visualization, object detection, segmentation, and autoencoders.

> 📅 **Deadline:** All assignments follow the schedule as per CSE4261 instructions.

---

## 📁 Folder Structure

Each folder corresponds to a specific assignment:

Assignment-1/
Assignment-2/
Assignment-3/
Assignment-4/
Assignment-5/
Assignment-6/
Assignment-7/
Assignment-8/
...


---

## 📌 Assignments Overview

### ✅ Assignment 1: 📊 CNN Comparison on CIFAR-100
- Compare performance of 10 pre-trained CNNs using 20 CIFAR-100 classes.
- Metrics: Accuracy, Model Size, Inference Time, Architecture.
- 📄 Hand-written report due via CR, and code + digital report via GitHub.

---

### ✅ Assignment 2: ⚙️ Activation Functions & Kernel Types
- Analyze effects of different activation functions on 10 CNNs.
- Discuss 3 CNNs using:
  - ✅ Regular / Deformable / Dilated Kernels
  - ✅ Depthwise / Pointwise / Modified Depthwise-Separable Kernels
- Visualize feature maps from various layers.

---

### ✅ Assignment 3: 🔍 Feature Extraction Before & After Transfer Learning
- Use your favorite CNN pretrained on ImageNet.
- Transfer learn on MNIST dataset.
- Project feature vectors to 2D using:
  - PCA
  - t-SNE
  - UMAP

---

### ✅ Assignment 4: 🧮 Manual Neural Network + Backpropagation
- Manually draw:
  - Neural Network (3 hidden layers)
  - Forward & Backpropagation graphs
- Derive weight update equations.
- Train using:
  - `tf.GradientTape()`
  - `model.fit()`
- Compare both training strategies.

---

### ✅ Assignment 5: ⚔️ Adversarial Attack with FGSM
- Implement **Fast Gradient Sign Method** (FGSM).
- Test on ImageNet class image.
- Discuss robustness under:
  - FGSM Noise
  - Gaussian Noise

🔗 [FGSM Tutorial (TensorFlow)](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)  
🔗 [ImageNet Class Map](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)

---

### ✅ Assignment 6: 🔦 Explainable AI — Grad-CAM & Integrated Gradients
- Visualize important image regions with:
  - Grad-CAM
  - Integrated Gradients
- Compare:
  - Original vs. Adversarial input activations
  - Effect of choosing Softmax vs. Pre-Softmax layers

---

### ✅ Assignment 7: 📹 YOLOv8-v11-v12 Object Detection & Face Detection
- Detect objects from a human-free video using:
  - YOLOv8, YOLOv11, YOLOv12 (Ultralytics)
- Fine-tune YOLOv8 on WIDER FACE dataset.
- Compare with:
  - [Yusepp/YOLOv8-Face](https://github.com/Yusepp/YOLOv8-Face)
- Build & train a YOLOv1-based face detector.

---

### ✅ Assignment 8: 👥 Crowd Counting and Segmentation
- Train & evaluate:
  - U-Net on segmentation dataset
  - U-Net & MCNN on crowd counting datasets
- Compare performance between U-Net and MCNN.

---

## 🚧 Future Work

- ✅ Add visualizations and plots per assignment.
- ✅ Add performance tables (accuracy, loss, etc.).
- ✅ Link Colab notebooks or Jupyter examples (if available).

---

## 🔗 GitHub Link

👉 [GitHub Repository](https://github.com/Arif111866/Deep-Learning-AI.git)

---

## 🤝 License

This repository is for academic purposes only (CSE4261 - Deep Learning). Please do not plagiarize.

---
