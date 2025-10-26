# Install dependencies if not already installed
# pip install streamlit sentence-transformers scikit-learn

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

projects = [
    "Designing a CNN-based system for thermal image analysis",
    "Development of IoT-based home automation system using microcontrollers",
    "Natural Language Processing for sentiment analysis of social media",
    "CFD simulations for airflow optimization in HVAC systems"
]

project_embeddings = model.encode(projects)

st.title("🎓 Master Project Recommendation Chatbot")
st.write("Enter your background, skills, or research interests, and get personalized project suggestions!")

user_input = st.text_input("Your Background / Skills:")

if user_input:
    student_embedding = model.encode([user_input])

    similarities = cosine_similarity(student_embedding, project_embeddings)[0]

    top_n = 3
    top_indices = np.argsort(similarities)[::-1][:top_n]

    st.write(f"**Top {top_n} Recommended Projects:**")
    for idx in top_indices:
        st.write(f"- {projects[idx]} (Similarity: {similarities[idx]:.2f})")

    st.write("\n**Why these projects?**")
    for idx in top_indices:
        st.write(f"- '{projects[idx]}' is recommended because it closely matches your skills and interests.")
