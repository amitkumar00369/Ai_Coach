import faiss
import numpy as np
import pickle
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

# -----------------------------
# Load CLIP
# -----------------------------
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -----------------------------
# Dataset
# -----------------------------
data = [

    # ---------------- EXERCISES ----------------
    {
        "type": "exercise",
        "name": "Bench Press",
        "goal": "muscle_gain",
        "description": "chest strength training using barbell",
        "muscle_group": "chest",
        "difficulty": "beginner",
        "image": "https://example.com/images/bench_press.jpg",
        "video": "https://youtube.com/shorts/Xx5SHfyukek"
    },
    {
        "type": "exercise",
        "name": "Push-up",
        "goal": "muscle_gain",
        "description": "bodyweight chest exercise",
        "muscle_group": "chest",
        "difficulty": "beginner",
        "image": "https://example.com/images/pushup.jpg",
        "video": "https://youtube.com/shorts/pushup123"
    },
    {
        "type": "exercise",
        "name": "Squats",
        "goal": "muscle_gain",
        "description": "lower body strength exercise",
        "muscle_group": "legs",
        "difficulty": "beginner",
        "image": "https://example.com/images/squats.jpg",
        "video": "https://youtube.com/shorts/squat123"
    },

    # ---------------- RECIPES ----------------
    {
        "type": "recipe",
        "name": "Paneer Salad",
        "goal": "muscle_gain",
        "description": "high protein vegetarian meal",
        "calories": 350,
        "protein": 20,
        "diet": "veg",
        "image": "https://example.com/images/paneer_salad.jpg",
        "video": "https://youtube.com/shorts/phRtlOG0Ry0"
    },
    {
        "type": "recipe",
        "name": "Oats with Milk",
        "goal": "muscle_gain",
        "description": "high protein breakfast",
        "calories": 300,
        "protein": 12,
        "diet": "veg",
        "image": "https://example.com/images/oats.jpg",
        "video": "https://youtube.com/shorts/oats123"
    },
    {
        "type": "recipe",
        "name": "Dal Rice",
        "goal": "muscle_gain",
        "description": "protein rich Indian meal",
        "calories": 400,
        "protein": 15,
        "diet": "veg",
        "image": "https://example.com/images/dal_rice.jpg",
        "video": "https://youtube.com/shorts/dal123"
    },

    # ---------------- WORKOUT PLANS ----------------
    {
        "type": "workout",
        "name": "Beginner Muscle Gain Plan",
        "goal": "muscle_gain",
        "description": "full body workout plan",
        "duration": 45,
        "image": None,
        "video": None
    },

    # ---------------- NUTRITION ----------------
    {
        "type": "nutrition",
        "name": "High Protein Diet",
        "goal": "muscle_gain",
        "description": "balanced diet for muscle growth",
        "protein_target": "80g/day",
        "image": None,
        "video": None
    }
]

# -----------------------------
# Encoding Functions (FIXED)
# -----------------------------
def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)

    outputs = model.get_text_features(**inputs)

    # 🔥 HANDLE DIFFERENT RETURN TYPES
    if hasattr(outputs, "pooler_output"):
        outputs = outputs.pooler_output

    vec = outputs.detach().numpy()[0]

    # Normalize
    vec = vec / np.linalg.norm(vec)

    return vec


def encode_image(url):
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        outputs = model.get_image_features(**inputs)

        if hasattr(outputs, "pooler_output"):
            outputs = outputs.pooler_output

        vec = outputs.detach().numpy()[0]

        # Normalize
        vec = vec / np.linalg.norm(vec)

        return vec
    except:
        return None


# -----------------------------
# Build Index
# -----------------------------
vectors = []
metadata = []

for item in data:
    text_input = f"{item['name']} {item['goal']} {item['description']}"
    text_vec = encode_text(text_input)

    vectors.append(text_vec)
    metadata.append(item)

    if item["image"]:
        img_vec = encode_image(item["image"])
        if img_vec is not None:
            vectors.append(img_vec)
            metadata.append(item)

vectors = np.array(vectors).astype("float32")

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, "index/faiss.index")

with open("index/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Index built successfully")