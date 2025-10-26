# Command-line version - Interactive Chatbot

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

projects = [
    "Designing a CNN-based system for thermal image analysis",
    "Development of IoT-based home automation system using microcontrollers",
    "Natural Language Processing for sentiment analysis of social media",
    "CFD simulations for airflow optimization in HVAC systems"
]

project_embeddings = model.encode(projects)

print("\n" + "="*60)
print("🎓 Master Project Recommendation Chatbot")
print("="*60)
print("Enter your background, skills, or research interests")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Your Background / Skills: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nThank you for using the Recommendation Chatbot!")
        break
    
    if not user_input:
        print("Please enter some information about your background or skills.\n")
        continue
    
    student_embedding = model.encode([user_input])
    similarities = cosine_similarity(student_embedding, project_embeddings)[0]
    
    top_n = 3
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    print(f"\n✅ Top {top_n} Recommended Projects for You:")
    print("-" * 60)
    for idx in top_indices:
        print(f"\n{projects[idx]}")
        print(f"   Similarity Score: {similarities[idx]:.2f}")
    
    print("\n" + "="*60 + "\n")

