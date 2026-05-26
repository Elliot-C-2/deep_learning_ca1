# EMNIST Handwritten Letters Classification

This project focuses on building and optimizing Deep Learning models to classify handwritten letters using the EMNIST dataset. The dataset consists of 28x28 grayscale images mapping to 26 distinct classes (A through Z), merging uppercase and lowercase letters into single categories.
Project Objective

The goal was to design, train, and refine a Convolutional Neural Network (CNN) capable of identifying handwritten letters, evaluating the impact of data augmentation, normalization, and network architecture on reducing overfitting and improving test accuracy.
Key Decisions Made
1. Choosing CNNs
Chose to focus on Convolutional Neural Networks rather than standard dense networks. Because image classification relies heavily on spatial patterns, CNNs allow the model to automatically learn features like edges, curves, and textures through filters without losing pixel-relationship context.

2. Crucial Data Transposition and Cleaning
During initial exploration, we discovered that the EMNIST dataset structure requires a specific reshape and transpose operation to display the images right-side up. Implementing this correction proved critical; without it, the models were forced to train on mirrored letters, resulting in a severe overfitting gap (~0.25). Correcting this alignment immediately shrank the overfitting gap down to roughly ~0.08.

3. Implementing Hyperparameter and Regularization Callbacks

Instead of guessing when to stop training, we incorporated Keras callbacks to make the training process more robust:

    - EarlyStopping: Used to halt execution as soon as validation performance plateaued, saving compute time and automatically restoring the best model weights.

    - ReduceLROnPlateau: Allowed the learning rate to dynamically drop (e.g., scaling down from 1e-3 to 3.13e-5). This gave the model the ability to make much finer weight adjustments as it neared convergence.
