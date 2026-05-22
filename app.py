"""
Skin Lesion Classifier - Railway + Hugging Face Version
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import traceback
from huggingface_hub import hf_hub_download

# Device
device = torch.device("cpu")

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Global model
model = None


def load_model():
    """Load model once from Hugging Face"""
    global model

    if model is not None:
        return model

    try:
        print("🚀 Loading model from Hugging Face...")

        repo_id = "77omaryalova/skin-lesion-classifier"

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.pth",
            token=os.environ.get("HF_TOKEN")  # required if private repo
        )

        print("📦 Model downloaded to:", model_path)

        # Create model architecture (must match training)
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(512, 2)

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        m.load_state_dict(state_dict)

        m.to(device)
        m.eval()

        model = m

        print("✅ Model loaded successfully!")
        return model

    except Exception:
        print("❌ MODEL LOADING FAILED:")
        print(traceback.format_exc())
        return None


def predict(image):
    """Predict lesion type"""

    if image is None:
        return "❌ Please upload an image"

    m = load_model()

    if m is None:
        return "❌ Model failed to load. Check logs (HF repo / token / filename)."

    try:
        img = image.convert("RGB")

        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = m(x)
            probs = F.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        label = "🔴 MELANOMA" if pred.item() == 1 else "🟢 BENIGN"
        confidence_pct = confidence.item() * 100

        return f"{label}\n\nConfidence: {confidence_pct:.2f}%"

    except Exception:
        return "❌ Prediction error:\n" + traceback.format_exc()


# Load model at startup (important for Railway stability)
model = load_model()

# Gradio UI
with gr.Blocks(title="Skin Lesion Classifier", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔬 Skin Lesion Classifier

    Upload a dermoscopy image and the AI will classify it as:

    - 🟢 Benign  
    - 🔴 Melanoma  

    ⚠️ Educational use only — not medical diagnosis.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")

        with gr.Column():
            output = gr.Textbox(label="Prediction")

    btn = gr.Button("🔍 Predict", variant="primary")
    btn.click(predict, inputs=image_input, outputs=output)

    gr.Markdown("""
    ---
    *Model:* ResNet18  
    *Platform:* Railway + Hugging Face  
    """)


if _name_ == "_main_":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8000))
    )
