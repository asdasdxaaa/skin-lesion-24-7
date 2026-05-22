"""
Skin Lesion Classifier - Railway.app Version with Hugging Face Model
=====================================================================
Dermoscopy Skin Lesion Classification
Group: AHMED AYASH, OMAR ADEL MOHAMMED, AYAH RAJOUB, GHADA FRIGUI
Institution: İstinye Üniversitesi
Year: 2026
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
from huggingface_hub import hf_hub_download

# Device
device = torch.device("cpu")

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Global model
model = None

def load_model():
    """Load your trained model from Hugging Face."""
    global model
    
    if model is not None:
        return model
    
    try:
        print("Loading model from Hugging Face...")
        
        # Download model from Hugging Face
        model_path = hf_hub_download(
            repo_id="77omaryalova/skin-lesion-classifier",
            filename="model.pth"
        )
        
        print(f"Model downloaded to: {model_path}")
        
        # Load the model
        m = models.resnet18(pretrained=False)
        m.fc = nn.Linear(512, 2)
        
        m.load_state_dict(torch.load(model_path, map_location=device))
        m.to(device)
        m.eval()
        
        model = m
        print("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("❌ model.pth not found on Hugging Face!")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict(image):
    """Predict Benign or Melanoma from image."""
    
    if image is None:
        return "❌ Please upload an image"
    
    # Load model on first use
    m = load_model()
    
    if m is None:
        return "❌ Model not loaded. Check Hugging Face repo or internet connection."
    
    try:
        # Ensure RGB
        img = image.convert("RGB")
        
        # Preprocess
        x = preprocess(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logits = m(x)
            probs = F.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
        
        # Results
        pred_class = pred_class.item()
        confidence_pct = confidence.item() * 100
        
        label = "🔴 MELANOMA" if pred_class == 1 else "🟢 BENIGN"
        
        result = f"{label}\n\nConfidence: {confidence_pct:.2f}%"
        return result
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create interface
with gr.Blocks(theme=gr.themes.Soft(), title="Skin Lesion Classifier") as demo:
    gr.Markdown("""
    # 🔬 Skin Lesion Classifier
    
    Upload a **dermoscopy image** and our ResNet18 AI model will predict if it's **Benign** or **Melanoma**.
    
    ⚠️ **Disclaimer:** For research and educational purposes only. Not a substitute for professional medical diagnosis.
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Dermoscopy Image")
        
        with gr.Column():
            result_output = gr.Textbox(label="Prediction Result", interactive=False)
    
    # Button
    analyse_btn = gr.Button("🔍 Analyse Image", variant="primary", size="lg")
    analyse_btn.click(predict, inputs=image_input, outputs=result_output)
    
    gr.Markdown("""
    ---
    **Project:** Dermoscopy Skin Lesion Classification  
    **Group Members:** AHMED AYASH, OMAR ADEL MOHAMMED, AYAH RAJOUB, GHADA FRIGUI  
    **Institution:** İstinye Üniversitesi  
    **Year:** 2026  
    **Model:** ResNet18 with Transfer Learning  
    **Dataset:** PH2 Dermoscopy Dataset
    """)

if __name__ == "__main__":
    # Launch on Railway
    demo.launch(server_name="0.0.0.0", server_port=8000)
