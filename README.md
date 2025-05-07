# GroceryNet: Web-based Application for Simultaneous Segmentation and Classification of Grocery Product Images
<p align="center">
  <img src="https://github.com/user-attachments/assets/d57cba90-5b8f-4c1d-b7b4-2dfa1c5ce42c" width="300">
</p>

GroceryNet is a web-based application for grocery product classification and segmentation using a dual-input parallel CNN model. It leverages multicolor spaces and pre-trained ResNeXt50_32x4d encoders to achieve accurate results. This project was developed as part of a thesis titled **"Grocery Product Image Analysis using Multicolor Dual-Input Parallel CNN for Simultaneous Semantic Segmentation and Classification"**.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset and Model](#dataset-and-model)
6. [Download Links](#download-links)
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
- Python 3.8 or higher (https://www.python.org/downloads/)
- pip (Python Package Manager)
- Jupyter Notebook (optional, for going through how the model was trained)
- Git (optional, for cloning the repository)
- Streamlit (https://streamlit.io/)
- A code editor (e.g., VSCode or PyCharm)

### Steps
1. **Clone the Repository (Optional)**: 
   If you have not already downloaded the project files, you can clone the repository, where the “your-username” is your github username:

   ```bash 
   git clone https://github.com/your-username/GroceryNet.git 
   cd GroceryNet

  Alternatively, if the files were manually downloaded, ensure you are in the correct project directory.


2. **Set Up a Virtual Environment (Optional but Recommended)**:
   To avoid package conflicts, create and activate a virtual environment:

   Windows:
   ```bash
    python -m venv venv          # for Mac/Linux: python3 -m venv venv
    venv\Scripts\activate        # source venv/bin/activate
  

3. **Download the Model and Dataset**:
- Download the pre-trained model (best_run.pth) and dataset from the Google Drive link.

- Place the best_run.pth file in the root directory of the project.

- Extract the dataset and place it in the appropriate directory (e.g., Hierarchical Grocery Store Dataset with Masks).

## Usage
- **Run the Application**: streamlit run "app.py"
Example:
![image](https://github.com/user-attachments/assets/a21192dc-db7c-4206-a970-7d825796c566)

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
- **Model**: Download "best_run.pth" [https://drive.google.com/file/d/1Ew-54l0w0p4EDOTzy-urv_eaVSQ-wP2s/view?usp=sharing](https://drive.google.com/file/d/1Ew-54l0w0p4EDOTzy-urv_eaVSQ-wP2s/view?usp=sharing)

- **Dataset**: Download and extract "Hierarchical Grocery Store Dataset with Masks.zip" [https://drive.google.com/file/d/1swV1AqYNt6miggjxV9pImcS4W_Ef8DB9/view?usp=sharing](https://drive.google.com/file/d/1swV1AqYNt6miggjxV9pImcS4W_Ef8DB9/view?usp=sharing)

- **Model Training**: Download "GroceryNetImplementation.ipynb" [https://drive.google.com/file/d/1w1wdFu67Zuy9zPtBJrKUnmYyXjSt-mjN/view?usp=sharing](https://drive.google.com/file/d/1w1wdFu67Zuy9zPtBJrKUnmYyXjSt-mjN/view?usp=sharing)
Note: The "Model Training" link is provided for those who want to go through how the model was trained.

## GUI of GroceryNet
![image](https://github.com/user-attachments/assets/fa892dcf-995c-480b-99da-8385b0535e01)
![image](https://github.com/user-attachments/assets/a31418a7-0ac1-4083-9a42-7f42122c3df8)


## Acknowledgments
[University of the Philippines Tacloban College](https://www.uptacloban.edu.ph/): For providing the resources and support for this project.

- **Streamlit**: For the easy-to-use web framework.

- **PyTorch**: For the deep learning framework.

- **Segmentation Models PyTorch**: For the U-Net implementation.

