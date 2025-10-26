# Install required library (if not already installed)
# !pip install sentence-transformers scikit-learn

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')


students = [
    "Python, Machine Learning, Signal Processing, Deep Learning",
    "Embedded Systems, Microcontrollers, IoT, C++"
]

projects = [
    "Designing a CNN-based system for thermal image analysis",
    "Development of IoT-based home automation system using microcontrollers",
    "Natural Language Processing for sentiment analysis of social media",
    "CFD simulations for airflow optimization in HVAC systems"
]

student_embeddings = model.encode(students)
project_embeddings = model.encode(projects)

def recommend_projects(student_index, top_n=2):
    student_emb = student_embeddings[student_index].reshape(1, -1)
    similarities = cosine_similarity(student_emb, project_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    print(f"Top {top_n} recommended projects for Student {student_index+1}:")
    for idx in top_indices:
        print(f"- {projects[idx]} (Similarity: {similarities[idx]:.2f})")

recommend_projects(student_index=0, top_n=2)  # Recommendations for first student
recommend_projects(student_index=1, top_n=2)  # Recommendations for second student
