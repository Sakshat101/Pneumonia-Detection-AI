Pneumonia Detection from Chest X-Rays
This project implements an end-to-end deep learning pipeline to identify pneumonia in chest X-ray images. It utilizes Transfer Learning with the MobileNetV2 architecture and provides a web-based interface for real-time inference using Streamlit.

Project Structure
train.py: Script for data augmentation, model training, and performance evaluation.

app.py: Streamlit application for uploading images and viewing model predictions.

pneumonia_model.h5: The saved weights and architecture of the trained model (generated after training).

Technical Specifications
1. Data Pipeline
The system is designed to work with the Chest X-Ray Images (Pneumonia) dataset. The preprocessing pipeline includes:

Normalization: Pixel values rescaled to a [0, 1] range.

Augmentation: Random rotations, width/height shifts, shear, zoom, and horizontal flips to prevent overfitting.

Target Size: All images are resized to 224x224 pixels.

2. Model Architecture
Base Model: MobileNetV2 (Pre-trained on ImageNet, frozen during initial training).

Custom Head:

Global Average Pooling 2D.

Dense layer (256 units, ReLU activation).

Dropout layer (0.5) for regularization.

Final Output layer (1 unit, Sigmoid activation) for binary classification.

Optimizer: Adam (Learning rate = 0.001).

Loss Function: Binary Crossentropy.

3. Requirements
Ensure the following Python libraries are installed:

TensorFlow

Streamlit

NumPy

Pillow

Matplotlib

Scikit-learn

Seaborn

Setup and Usage
Step 1: Install Dependencies
Run the following command to install the necessary packages:

Bash

pip install tensorflow streamlit numpy pillow matplotlib scikit-learn seaborn
Step 2: Model Training
Before running the web application, you must train the model to generate the .h5 file. Ensure your dataset paths are correctly configured in the script.

Bash

python3 train.py
This script will output the training/validation accuracy plots and a confusion matrix upon completion.

Step 3: Launching the Web Tool
Once the pneumonia_model.h5 file has been generated, launch the Streamlit interface:

Bash

python3 -m streamlit run app.py
Disclaimer
This tool is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis. Always consult with a qualified healthcare professional for medical advice and interpretation of clinical images.
