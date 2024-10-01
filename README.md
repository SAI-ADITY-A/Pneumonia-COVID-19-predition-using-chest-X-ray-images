# Pneumonia Prediction Using Chest X-Ray Images
## Overview
This project aims to develop a deep learning-based image classification model using ResNet architecture to predict pneumonia from chest X-ray images. The dataset consists of chest X-ray images classified into three categories: COVID19, NORMAL, and PNEUMONIA. We trained a ResNet model on the dataset and evaluated its performance using various metrics such as accuracy, precision, recall, and confusion matrix.

## Dataset
The dataset used for this project is downloaded from Kaggle: Chest X-ray COVID19 & Pneumonia. It consists of X-ray images organized into two folders: train and test, with the following classes:

COVID19: X-rays of patients diagnosed with COVID19.

NORMAL: X-rays of healthy individuals.

PNEUMONIA: X-rays of patients diagnosed with pneumonia (non-COVID).

## Project Workflow
### 1. Exploratory Data Analysis (EDA)
Loaded and visualized the dataset to understand the distribution of images across classes.
Displayed random samples from each category to observe visual differences.
### 2. Preprocessing
Applied data augmentation techniques such as rotation, zoom, and horizontal flip to prevent overfitting and enhance generalization.
Rescaled image pixel values to the range [0, 1] using ImageDataGenerator.
### 3. Model Architecture (ResNet50)
We employed the ResNet50 architecture, which is pre-trained on the ImageNet dataset, as the base model.
Fine-tuned the model by adding custom layers for the chest X-ray classification task.
Key Layers:
Global Average Pooling
Fully Connected Dense Layer (512 units)
Output Layer with Softmax activation (3 units for the 3 classes)
### 4. Training
The model was trained using the Adam optimizer with a learning rate of 1e-4.
We used the categorical cross-entropy loss function since this is a multi-class classification problem.
Validation was performed using the test set, and early stopping was applied to prevent overfitting.
### 5. Evaluation
After training, the model was evaluated using various metrics, including:
Accuracy
Precision
Recall
Confusion Matrix
A classification report was generated to display the detailed performance of the model across the three classes.
### 6. Visualization
Confusion matrix plotted to understand how well the model performed in differentiating between the three classes.
Accuracy and loss curves for training and validation were plotted to visualize model performance over epochs.
Confusion matrix plotted to understand how well the model performed in differentiating between the three classes.
Accuracy and loss curves for training and validation were plotted to visualize model performance over epochs.
