import faiss
import numpy as np
import pickle
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

# ✅ FIX IMPORT (use correct module)
from datasets import recipes, running, nutrition, exercises

# -----------------------------
# Load CLIP
# -----------------------------
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -----------------------------
# Dataset
# -----------------------------
data = recipes + running + nutrition + exercises


# -----------------------------
# Ensure index folder exists
# -----------------------------
os.makedirs("index", exist_ok=True)


# -----------------------------
# Encoding Functions
# -----------------------------
def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)

    outputs = model.get_text_features(**inputs)

    # Handle different output formats
    if hasattr(outputs, "pooler_output"):
        outputs = outputs.pooler_output

    vec = outputs.detach().numpy()[0]

    # Normalize
    vec = vec / np.clip(np.linalg.norm(vec), 1e-10, None)

    return vec


# ⚡ OPTIONAL: disable image for faster build
USE_IMAGE = False


def encode_image(url):
    if not USE_IMAGE:
        return None

    try:
        response = requests.get(url, timeout=3)
        image = Image.open(response.raw).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model.get_image_features(**inputs)

        if hasattr(outputs, "pooler_output"):
            outputs = outputs.pooler_output

        vec = outputs.detach().numpy()[0]
        vec = vec / np.clip(np.linalg.norm(vec), 1e-10, None)

        return vec

    except Exception as e:
        print("Image error:", e)
        return None


# -----------------------------
# Build Index
# -----------------------------
def build_data():

    print("🚀 Building index...")

    vectors = []
    metadata = []

    for i, item in enumerate(data):

        # Add dummy media
        item["image"] = f"https://dummyimage.com/300x300&text={item['name']}"
        item["video"] = "https://youtube.com/shorts/demo"

        # Text embedding
        text_input = f"{item['name']} {item.get('goal','')} {item.get('description','')}"
        text_vec = encode_text(text_input)

        vectors.append(text_vec)
        metadata.append(item)

        # Image embedding (optional)
        img_vec = encode_image(item["image"])
        if img_vec is not None:
            vectors.append(img_vec)
            metadata.append(item)

        print(f"✅ Processed {i+1}/{len(data)}")

    # Convert to numpy
    vectors = np.array(vectors).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save
    faiss.write_index(index, "index/faiss.index")

    with open("index/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Index built successfully")

    return {"message": "Index built successfully"}

# print(build_data())