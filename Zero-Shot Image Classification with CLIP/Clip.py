from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch


dataset = load_dataset("rajuptvs/ecommerce_products_clip", split="train")

image = dataset[999]["image"]


categories = ["shirt", "trousers", "shoes", "dress", "hat",
              "bag", "watch", "glasses", "jacket", "belt"]

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

inputs = processor(
    text=categories,
    images=image,
    return_tensors="pt",
    padding=True
)
outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=1)

predicted_index = probs.argmax().item()
predicted_category = categories[predicted_index]

print("Predicted Category:", predicted_category)
print("Probabilities per category:")
for cat, prob in zip(categories, probs[0]):
    print(f"{cat}: {prob.item():.4f}")
