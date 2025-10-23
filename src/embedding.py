# src/embedding.py
from db import get_collection
from sentence_transformers import SentenceTransformer
import torch

# --- Obtener colección ---
COLLECTION_NAME = "sitios"
places_col = get_collection(COLLECTION_NAME)  # ya no se llama como función, get_collection devuelve la colección

# --- Cargar modelo ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# --- Cargar datos de Mongo ---
corpus_sitios = list(places_col.find({}))
corpus_texts = [
    (s.get("nombre") or "") + " " + (s.get("direccion") or "")
    for s in corpus_sitios
]

# --- Generar embeddings ---
corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, device=DEVICE)
print(f"Embeddings generados para {len(corpus_embeddings)} sitios.")
