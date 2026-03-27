from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline

app = FastAPI()

# -----------------------------
# Load Models
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# -----------------------------
# Load Vector DB
# -----------------------------
index = faiss.read_index("index/faiss.index")

with open("index/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# -----------------------------
# User Input
# -----------------------------
class UserInput(BaseModel):
    goal: str
    query: str
    age: int
    weight: float
    diet: str
    level: str
    time_available: int

# -----------------------------
# Encode Query
# -----------------------------
def encode_query(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = clip_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        features = outputs.pooler_output

    vec = features.cpu().numpy()
    vec = vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-10, None)

    return vec.astype("float32")

# -----------------------------
# Pre-filter
# -----------------------------
def pre_filter(metadata, user):
    return {i for i, item in enumerate(metadata) if item.get("goal") == user.goal}

# -----------------------------
# Re-ranking
# -----------------------------
def rerank(items, user):
    def score(item):
        s = 0

        if item.get("goal") == user.goal:
            s += 3

        if item.get("type") == "exercise":
            s += 2

        if item.get("type") == "recipe" and item.get("diet") in [None, user.diet]:
            s += 2

        if user.level.lower() in item.get("description", "").lower():
            s += 1

        return s

    return sorted(items, key=score, reverse=True)

# -----------------------------
# Validation (Input)
# -----------------------------
def validate_user(user):
    if user.goal == "fat_loss" and user.weight < 40:
        raise ValueError("Unsafe plan")

    if user.time_available < 10:
        raise ValueError("Too little time")

# -----------------------------
# CORE PLAN BUILDER (IMPORTANT)
# -----------------------------
def build_base_plan(user, exercises, meals):
    plan = []

    for i in range(7):
        day_plan = {
            "day": i + 1,
            "workout": exercises[i % len(exercises)]["name"] if exercises else "Rest",
            "meal": meals[i % len(meals)]["name"] if meals else "Balanced Diet",
        }

        # Difficulty logic
        if user.level == "beginner":
            day_plan["sets"] = "2-3 sets"
        else:
            day_plan["sets"] = "4-5 sets"

        # Time logic
        day_plan["duration"] = f"{user.time_available} min"

        plan.append(day_plan)

    return plan

# -----------------------------
# AI REASONING (LLM USED CORRECTLY)
# -----------------------------
def enhance_plan_with_llm(plan, user):
    for day in plan:
        prompt = f"""
Give 1 short fitness tip.

Workout: {day['workout']}
Goal: {user.goal}
Level: {user.level}
"""

        result = llm(prompt, max_new_tokens=30, do_sample=False)
        tip = result[0]["generated_text"].strip()

        day["tip"] = tip

    return plan

# -----------------------------
# VALIDATION LAYER
# -----------------------------
def validate_final_plan(plan, user):
    for day in plan:
        if user.level == "beginner" and "5 sets" in day.get("sets", ""):
            day["sets"] = "3 sets"

        if user.goal == "fat_loss" and user.weight < 45:
            day["tip"] = "Consult a trainer before intense workouts"

    return plan

# -----------------------------
# ADD NUTRITION
# -----------------------------
def add_nutrition(plan):
    for day in plan:
        day["calories"] = 2000
        day["protein"] = "80g"
    return plan

# -----------------------------
# API
# -----------------------------
@app.post("/ai-coach")
async def ai_coach(user: UserInput):
    try:
        validate_user(user)

        # Encode query
        query_vec = encode_query(user.query)

        # Search
        D, I = index.search(query_vec, k=30)

        # Filter
        valid_ids = pre_filter(metadata, user)
        results = [metadata[i] for i in I[0] if i in valid_ids]

        if not results:
            return {"message": "No relevant data found"}

        # Re-rank
        results = rerank(results, user)

        # Split
        exercises = [r for r in results if r.get("type") == "exercise"][:3]
        meals = [r for r in results if r.get("type") == "recipe"][:3]

        # -----------------------------
        # FINAL PIPELINE
        # -----------------------------

        # 1. Build plan (backend logic)
        plan = build_base_plan(user, exercises, meals)

        # 2. Add AI reasoning
        plan = enhance_plan_with_llm(plan, user)

        # 3. Validate
        plan = validate_final_plan(plan, user)

        # 4. Nutrition
        plan = add_nutrition(plan)

        return {
            "user": user.dict(),
            "workout_items": exercises,
            "meal_items": meals,
            "weekly_plan": plan
        }

    except Exception as e:
        return {"error": str(e)}