# MNIST CNN 🧠

A simple **Convolutional Neural Network** built in **TensorFlow** to classify handwritten digits from the **MNIST dataset**.


- **main branch** → runs on **CPU**  
- **cuda branch** → runs on **GPU**  

---

## CUDA Branch Only:

You must have CUDA and cuDNN installed manually (not included in requirements.txt).

If you have an older GPU, you may encounter TensorFlow bugs. In that case, try:

```bash
pip install tensorflow==2.10.0
```

---
  
## 🚀 Overview

This project demonstrates how to construct, train, and evaluate a CNN for recognizing digits (0–9) in grayscale 28×28 pixel images. The MNIST dataset is one of the foundational benchmarks in deep learning and computer vision.

---

## ✍️ Interactive GUI

In addition to training and evaluating the model, the project includes an **interactive GUI** where you can **draw digits** and have them recognized by the trained CNN.  

---

## 🧱 Model Architecture

The network typically consists of:

- **Conv2D layer** → ReLU activation  
- **MaxPooling2D layer**  
- (Repeated stack of Conv + Pool)  
- **Flatten** → **Dense** → **Softmax** output for 10 classes  

This setup achieves robust performance (often 98–99% accuracy on test set) :contentReference[oaicite:0]{index=0}.

---

## 🛠️ How to Use

### Requirements

```bash
pip install tensorflow numpy matplotlib
```
---

### Training & Evaluation

```bash
python main.py
```
---

#### This will:

  1. Download and preprocess MNIST dataset

  2. Train the CNN model

  3. Evaluate test accuracy and print results

---

## 📊 Results

_Sample output when evaluation completes:_

```yaml
Test Accuracy: 0.99
```
_(Actual numbers depend on training parameters, epochs, and TensorFlow version.)_

---

## 📝 Optional Enhancements

**You might consider extending this project with:**

- Data augmentation (rotations, shifts)

- Additional Conv + Dense layers for deeper architectures

- Early stopping or checkpoint saving

- Visualizing training history using matplotlib

- Porting to TensorFlow 2.x + Keras or PyTorch

- Deploying as a web API or mobile app


## 📚 Learn More

Deep dive tutorial:
- [Machine Learning Mastery – Building a CNN for MNIST](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification)








\To run the application you need to execute the user_test.py file.

If you want to retrain your model you need to execute the train.py file.



This is for CUDA only:

You will need to have installed the CUDA & CUDNN version yourself!
They will not be included in the requirements!

You may experience some buggs with tensorflow if you have an old GPU,
if that happens I recommend you to try installing tensorflow==2.10.0 !
