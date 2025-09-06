from sentence_transformers import SentenceTransformer

# Correct Hugging Face model name
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",   # ✅ Correct path
    cache_folder="./models"
)
print("Model downloaded and cached successfully.")
