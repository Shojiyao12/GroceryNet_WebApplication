# GroceryNet: Web-based Application for Simultaneous Segmentation and Classification of Grocery Product Images
![GroceryNet_Web_Application_Logo](https://github.com/user-attachments/assets/15508d9e-0307-4e98-8801-36419d9c9a94)

GroceryNet is a web-based application for grocery product classification and segmentation using a dual-input parallel CNN model. It leverages multicolor spaces and pre-trained ResNeXt50_32x4d encoders to achieve accurate results. This project was developed as part of a thesis titled **"Grocery Product Image Analysis using Multicolor Dual-Input Parallel CNN for Simultaneous Semantic Segmentation and Classification"**.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset and Model](#dataset-and-model)
6. [Download Links](#download-links)
7. [Acknowledgement](#acknowledgement)

---

## Overview

GroceryNet is designed to classify and segment grocery product images using a deep learning model. The application provides:
- **Image Classification**: Predicts the class of a grocery product.
- **Semantic Segmentation**: Highlights the product in the image.
- **Top-5 Predictions**: Displays the top 5 probable classes with confidence scores.

The model is built using PyTorch and Streamlit for the web interface.

---

## Features

- **Dual-Input CNN**: Utilizes RGB and XYZ color spaces for improved accuracy.
- **Pre-Trained Encoders**: Uses ResNeXt50_32x4d encoders for feature extraction.
- **Interactive Web Interface**: Built with Streamlit for easy user interaction.
- **Top-5 Predictions**: Visualizes the top 5 predictions with confidence scores.
- **Segmentation Overlay**: Displays the segmentation mask over the input image.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. **Clone the Repository**:
   ```bash
    git clone https://github.com/your-username/GroceryNet.git
    cd GroceryNet

2. **Set Up a Virtual Environment (Optional but Recommended)**:
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**:
   ```bash
    pip install -r requirements.txt
   
4. **Download the Model and Dataset**:
- Download the pre-trained model (SHAWJIE_final_model_run5.pth) and dataset from the Google Drive link.

- Place the SHAWJIE_final_model_run5.pth file in the root directory of the project.

- Extract the dataset and place it in the appropriate directory (e.g., FINAL DATASET FOR FINAL DEFENSE).

## Usage
- **Run the Application**: streamlit run app.py
- **Upload an Image**: Use the file uploader in the web interface to upload a grocery product image (JPG or PNG format).

## View Results:

The application will display:

- The input image.

- The segmentation map overlaid on the image.

- The top-5 predictions with confidence scores.

## Dataset and Model
**Dataset**:
The dataset used for training and testing is the Hierarchical Grocery Store Dataset (Klasson et al., 2019). It contains images of grocery products along with their corresponding masks.

**Model**:
The model is a Parallel U-Net with pre-trained ResNeXt50_32x4d encoders. It was trained on the dataset mentioned above and achieves high accuracy in both classification and segmentation tasks.

## Download Links
- **Model**: Download "SHAWJIE_final_model_run5.pth" [https://drive.google.com/file/d/1swV1AqYNt6miggjxV9pImcS4W_Ef8DB9/view?usp=sharing](https://drive.google.com/file/d/1swV1AqYNt6miggjxV9pImcS4W_Ef8DB9/view?usp=sharing)

- **Dataset**: Download and extract "FINAL DATASET FOR FINAL DEFENSE.zip" [https://drive.google.com/file/d/1swV1AqYNt6miggjxV9pImcS4W_Ef8DB9/view?usp=sharing](https://drive.google.com/file/d/1swV1AqYNt6miggjxV9pImcS4W_Ef8DB9/view?usp=sharing)

- **FOUND HERE**: [https://drive.google.com/drive/u/4/folders/1FMTev5dG5ppSu_48APMlS8w9J2StLTFf](https://drive.google.com/drive/folders/1FMTev5dG5ppSu_48APMlS8w9J2StLTFf?usp=sharing)

## Acknowledgments
University of the Philippines Tacloban College: For providing the resources and support for this project.

- **Streamlit**: For the easy-to-use web framework.

- **PyTorch**: For the deep learning framework.

- **Segmentation Models PyTorch**: For the U-Net implementation.

