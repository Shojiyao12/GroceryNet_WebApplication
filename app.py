import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
import re
import segmentation_models_pytorch as smp
import torch.nn as nn
# Import the original code components

def standardize_class_name(folder_name):
    return re.sub(r'\s*\d+$', '', folder_name).strip()

def split_per_class(base_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Splits the dataset into training, validation, and test sets on a per-class basis.

    Args:
        base_dir (str): Base directory containing class folders.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: Lists of training, validation, and test data paths along with class to index mapping.
    """
    train_paths, val_paths, test_paths = [], [], []
    class_labels_set = set()
    class_to_folder_names = {}

    # Organize folder names per class
    for class_name in sorted(os.listdir(base_dir)):
        std_class_name = standardize_class_name(class_name)
        class_labels_set.add(std_class_name)
        class_to_folder_names.setdefault(std_class_name, []).append(class_name)

    # Create class to index mapping
    class_labels_list = sorted(class_labels_set)
    class_to_index = {label: idx for idx, label in enumerate(class_labels_list)}

    # Split data per class
    for std_class_name, folder_names in class_to_folder_names.items():
        images = []
        masks = []
        for folder_name in folder_names:
            image_folder = os.path.join(base_dir, folder_name, 'Images')
            mask_folder = os.path.join(base_dir, folder_name, 'Masks')
            if not os.path.exists(image_folder) or not os.path.exists(mask_folder):
                continue

            image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
            images_in_folder = [os.path.join(image_folder, f) for f in image_files]
            masks_in_folder = [os.path.join(mask_folder, os.path.splitext(f)[0] + '_mask.png') for f in image_files]

            images.extend(images_in_folder)
            masks.extend(masks_in_folder)

        class_indices = [class_to_index[std_class_name]] * len(images)
        data = list(zip(images, masks, class_indices))

        # Split data
        train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio))
        temp_ratio = val_ratio + test_ratio

        val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / temp_ratio))

        train_paths.extend(train_data)
        val_paths.extend(val_data)
        test_paths.extend(test_data)

    return train_paths, val_paths, test_paths, class_to_index
# Load the model and dataset
def load_model_and_data(base_dir):
    # Split the dataset
    train_paths, val_paths, test_paths, class_to_index = split_per_class(base_dir)
    
    # Create index_to_class mapping
    index_to_class = {v: k for k, v in class_to_index.items()}
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(cnn_encoder1="resnext50_32x4d", cnn_encoder2="resnext50_32x4d", 
                 out_seg_channels=1, num_classes=81, pretrained=True).to(device)
    state_dict = torch.load('SHAWJIE_final_model_run5.pth', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, test_paths, class_to_index, index_to_class, device

# Inference function
def inference(image_input, model, device, from_upload=False):
    if from_upload:
        image = cv2.imdecode(image_input, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_input)
    
    # Original preprocessing: convert to RGB, resize to 256x256, center crop to 224x224
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image_pil = Image.fromarray(image)
    image_pil = transforms.functional.center_crop(image_pil, (224, 224))
    image_np = np.array(image_pil)
    
    # Compute XYZ using OpenCV (like in your original pipeline)
    image_xyz = cv2.cvtColor(image_np, cv2.COLOR_RGB2XYZ)
    
    # Concatenate RGB and XYZ to form a 6-channel image
    image_combined = np.concatenate((image_np, image_xyz), axis=2)
    
    # Convert to tensor and normalize by dividing by 255
    image_tensor = torch.from_numpy(image_combined.transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        seg_output, cls_output = model(image_tensor)
    
    return image_np, seg_output, cls_output

class UNet(nn.Module):
    def __init__(self, cnn_encoder1="resnext50_32x4d", cnn_encoder2="resnext50_32x4d", 
                 out_seg_channels=1, num_classes=81, pretrained=False):
        super(UNet, self).__init__() 
        self.unet_cnn = smp.Unet(
            encoder_name=cnn_encoder1,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=out_seg_channels,
        )
        self.unet_vit = smp.Unet(
            encoder_name=cnn_encoder2,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=out_seg_channels,
        )
        self.cnn_encoder1_output_dim = self.unet_cnn.encoder.out_channels[-1]
        self.cnn_encoder2_output_dim = self.unet_vit.encoder.out_channels[-1]
        self.combined_output_dim = self.cnn_encoder1_output_dim + self.cnn_encoder2_output_dim
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classification_head = nn.Sequential(
            nn.Linear(self.combined_output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.45),  # Dropout added
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),  # Dropout added
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x_rgb = x[:, :3, :, :] 
        x_xyz = x[:, 3:, :, :]
        segmentation_output_cnn = self.unet_cnn(x_rgb)
        segmentation_output_vit = self.unet_vit(x_xyz)
        combined_segmentation_output = (segmentation_output_cnn + segmentation_output_vit) / 2
        cnn_encoder1_features = self.adaptive_pool(self.unet_cnn.encoder(x_rgb)[-1])
        cnn_encoder1_features = cnn_encoder1_features.view(cnn_encoder1_features.size(0), -1)
        cnn_encoder2_features = self.adaptive_pool(self.unet_vit.encoder(x_xyz)[-1])
        cnn_encoder2_features = cnn_encoder2_features.view(cnn_encoder2_features.size(0), -1)
        combined_features = torch.cat((cnn_encoder1_features, cnn_encoder2_features), dim=1)
        classification_output = self.classification_head(combined_features)
        return combined_segmentation_output, classification_output

# Visualization functions
def plot_overlay(image, seg_mask):
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.imshow(seg_mask, alpha=0.5, cmap='jet')
    plt.axis('off')
    return plt

def plot_top5(predictions, index_to_class):
    probs = F.softmax(predictions, dim=1)[0]
    top5_probs, top5_indices = torch.topk(probs, 5)
    df = pd.DataFrame({
        'Class': [index_to_class[i.item()] for i in top5_indices],
        'Probability': [f"{p.item()*100:.2f}%" for p in top5_probs]
    }, index=[f"{i+1}" for i in range(5)])  # Change the index to start from 1
    return df

# Function to plot Top-5 Predictions as a bar graph
def plot_top5_graph(top5_df):
    plt.figure(figsize=(10, 5))
    plt.barh(top5_df['Class'], top5_df['Probability'].str.rstrip('%').astype(float), color='red')
    plt.xlabel('Accuracy', color='white')
    plt.ylabel('Top-5 Predictions', color='white')
    plt.xlim(0, 100)
    plt.gca().invert_yaxis()
    plt.gca().set_facecolor('#0e1117')
    plt.gcf().patch.set_facecolor('#0e1117')
    plt.tick_params(colors='white')
    plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, 101, 20))  # Add gridlines at every 20%
    return plt

# Streamlit App
def main():
    # Adjust the sidebar to prevent shaking
    st.sidebar.image(r"C:\\Users\\Shaw Jie Yao\\Desktop\\College Files\\4th Year College UP 2024-2025\\2nd Semester Subjects\\CMSC 199.2 (Lec) - G\\GroceryNet_Web_Application_Logo.jpg", width=200)

    # Static sidebar content without dynamic updates
    st.sidebar.markdown("""
        ## GroceryNet
        
        **What is GroceryNet?**  
        GroceryNet is a web-based grocery product classification and segmentation system that uses multicolor spaces within a dual-input CNN.  
        
        **Model:** Parallel U-Net with Pre-Trained ResNeXt50_32x4d Encoders  
        
        **Dataset:** Hierarchical Grocery Store (Klasson et al., 2019)
    """, unsafe_allow_html=True)

    # Load model and data
    base_dir = r"C:\\Users\\Shaw Jie Yao\\Desktop\\College Files\\4th Year College UP 2024-2025\\2nd Semester Subjects\\CMSC 199.2 (Lec) - G\\FINAL DATASET FOR FINAL DEFENSE-20250124T023231Z-001\\FINAL DATASET FOR FINAL DEFENSE"
    model, test_paths, class_to_index, index_to_class, device = load_model_and_data(base_dir)
    
    # Allow user to upload an image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image, seg_output, cls_output = inference(file_bytes, model, device, from_upload=True)
        
        seg_mask = torch.sigmoid(seg_output).squeeze().cpu().numpy() > 0.5
        cls_pred = torch.argmax(cls_output).item()
        
        # Add Top-1 Prediction Display Here
        probs = F.softmax(cls_output, dim=1)[0]
        top1_prob, top1_idx = torch.max(probs, dim=0)
        top1_class = index_to_class[top1_idx.item()]

        st.markdown(
            f"""
            <div style='background-color: #2e7d32; padding: 10px; border-radius: 10px; color: white;'>
                <strong>Prediction:</strong> {top1_class}<br>
                <strong>Confidence (Top-1 Accuracy):</strong> {top1_prob.item() * 100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Segmentation Map")
            fig = plot_overlay(image, seg_mask)
            st.pyplot(fig)

        st.subheader("Top-5 Predictions")
        top5_df = plot_top5(cls_output, index_to_class)
        st.table(top5_df)

        # Add the Top-5 Predictions Graph here
        top5_graph = plot_top5_graph(top5_df)
        st.pyplot(top5_graph)

    else:
        st.warning("Please upload an image to proceed.")

if __name__ == "__main__":
    main()
