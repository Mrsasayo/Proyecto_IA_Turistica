# src/rag.py
# Versión optimizada: carga embeddings/FAISS desde disco si existen,
# evita ejecuciones pesadas en import y permite rebuild explícito vía script.

import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
import torch
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import faiss
import numpy as np
import pickle
import streamlit as st  # <-- añadido para obtener idioma desde sesión

# -------------------------------------------------------
# Configuración logging
# -------------------------------------------------------
log_file = 'rag_gemini.log'
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("Módulo RAG (optimizado) inicializado (sin cargas forzadas).")

# -------------------------------------------------------
# Cargar env
# -------------------------------------------------------
dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path=dotenv_path)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "turismo_cali")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sitios")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

# Paths para índices/embeddings
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
os.makedirs(DATA_DIR, exist_ok=True)
FAISS_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
EMB_MATRIX_PATH = os.path.join(DATA_DIR, "embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")

# Flag para rebuild controlado (por defecto False)
REBUILD_INDEX = os.getenv("REBUILD_INDEX", "false").lower() == "true"

# -------------------------------------------------------
# Conexión a MongoDB (no cargar corpus aquí)
# -------------------------------------------------------
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    places_col = db[COLLECTION_NAME]
    logging.info(f"Conectado a MongoDB: Base={DB_NAME}, Colección={COLLECTION_NAME}")
except Exception as e:
    logging.critical(f"No se pudo conectar a MongoDB: {e}")
    raise SystemExit(e)

# -------------------------------------------------------
# Lazy resources cached in module (will be created on demand)
# -------------------------------------------------------
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None            # SentenceTransformer instance
_faiss_index = None
_embeddings = None       # numpy matrix
_metadata = None         # list of dicts (corpus metadata)

def get_device():
    return _device

# -------------------------------------------------------
# Embedding model (multilingüe opcional)
# -------------------------------------------------------
def get_embedding_model(multilingual: bool = False):
    global _model
    if _model is None:
        # si multilingual True, usar modelo multilingüe
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" if multilingual else MODEL_NAME
        logging.info(f"Load pretrained SentenceTransformer: {model_name} on {get_device()}")
        _model = SentenceTransformer(model_name, device=get_device())
    return _model

def load_faiss_index():
    global _faiss_index, _embeddings, _metadata
    if _faiss_index is not None:
        return _faiss_index, _embeddings, _metadata

    if not REBUILD_INDEX and os.path.exists(FAISS_PATH) and os.path.exists(EMB_MATRIX_PATH) and os.path.exists(META_PATH):
        try:
            logging.info("🔁 Cargando índice FAISS y embeddings desde disco (data/)...")
            index = faiss.read_index(FAISS_PATH)
            embeddings = np.load(EMB_MATRIX_PATH)
            with open(META_PATH, "rb") as f:
                metadata = pickle.load(f)
            _faiss_index = index
            _embeddings = embeddings
            _metadata = metadata
            logging.info(f"FAISS cargado: {embeddings.shape[0]} vectores, dimensión {embeddings.shape[1]}")
            return _faiss_index, _embeddings, _metadata
        except Exception as e:
            logging.warning(f"No se pudo cargar FAISS desde disco: {e}. Se requerirá rebuild si se solicita.")
            return None, None, None
    return None, None, None

def save_faiss_index(index, embeddings, metadata):
    try:
        logging.info("💾 Guardando FAISS y embeddings en disco...")
        faiss.write_index(index, FAISS_PATH)
        np.save(EMB_MATRIX_PATH, embeddings)
        with open(META_PATH, "wb") as f:
            pickle.dump(metadata, f)
        logging.info("Guardado completado.")
    except Exception as e:
        logging.error(f"Error guardando índice/embeddings: {e}")

# -------------------------------------------------------
# Función para construir embeddings + FAISS
# -------------------------------------------------------
def build_index_from_db(batch_size=256, multilingual: bool = False):
    logging.info("Construyendo índice FAISS desde MongoDB (esto es costoso) ...")
    docs = list(places_col.find({}))
    logging.info(f"Se recuperaron {len(docs)} documentos de MongoDB.")
    texts = [
        f"{d.get('nombre','')}. {d.get('categoria','')}. {d.get('descripcion','')}. Dirección: {d.get('direccion','')}. Comentarios: {d.get('comentarios','')}"
        for d in docs
    ]
    model = get_embedding_model(multilingual=multilingual)
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        embs = model.encode(batch_texts, show_progress_bar=True, convert_to_numpy=True)
        all_embs.append(embs)
    all_embs = np.vstack(all_embs).astype('float32')
    dim = all_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(all_embs)
    index.add(all_embs)
    metadata = docs
    save_faiss_index(index, all_embs, metadata)
    return index, all_embs, metadata

# -------------------------------------------------------
# Búsqueda por FAISS / fallback si FAISS no existe
# -------------------------------------------------------
def buscar_sitios_relevantes(query: str, top_k: int = 5, multilingual: bool = False):
    index, embeddings, metadata = load_faiss_index()
    model = get_embedding_model(multilingual=multilingual)
    q_emb = model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)
    if index is not None:
        D, I = index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            doc = metadata[int(idx)]
            results.append({
                "score": float(score),
                "nombre": doc.get("nombre", "No disponible"),
                "direccion": doc.get("direccion", "No disponible"),
                "categoria": doc.get("categoria", "No disponible"),
                "descripcion": doc.get("descripcion", "No disponible"),
                "latitud": doc.get("latitud", None),
                "longitud": doc.get("longitud", None),
                "comentarios": doc.get("comentarios", "No hay comentarios")
            })
        return results
    else:
        logging.info("FAISS no disponible — usando búsqueda por similitud con SentenceTransformer (fallback).")
        texts = [f"{d.get('nombre','')}. {d.get('categoria','')}. {d.get('descripcion','')}" for d in list(places_col.find({}))]
        all_embs = model.encode(texts, convert_to_tensor=True)
        q_emb_t = model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb_t, all_embs)[0]
        topk = min(top_k, len(texts))
        top_results = torch.topk(cos_scores, k=topk)
        results = []
        docs = list(places_col.find({}))
        for score, idx in zip(top_results[0], top_results[1]):
            d = docs[int(idx)]
            results.append({
                "score": float(score.item()),
                "nombre": d.get("nombre","No disponible"),
                "direccion": d.get("direccion","No disponible"),
                "categoria": d.get("categoria","No disponible"),
                "descripcion": d.get("descripcion","No disponible"),
                "latitud": d.get("latitud", None),
                "longitud": d.get("longitud", None),
                "comentarios": d.get("comentarios", "No hay comentarios")
            })
        return results

# -------------------------------------------------------
# Gemini LLM wrapper
# -------------------------------------------------------
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logging.info("API de Google Gemini configurada (si está disponible).")
    except Exception as e:
        logging.error(f"Error config Gemini API: {e}")

def generar_respuesta_con_gemini(query: str, contexto: list, usuario_info: dict = None):
    if not GOOGLE_API_KEY:
        return "Error: Falta la clave API de Google. No se puede generar la respuesta."
    if not contexto:
        return "No se encontraron lugares relevantes para tu consulta."

    # Obtener idioma desde Streamlit session_state (fallback 'es')
    idioma_usuario: str = st.session_state.get("idioma_usuario", "es")

    try:
        llm_model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        logging.error(f"No se pudo inicializar el modelo Gemini '{GEMINI_MODEL}': {e}")
        return "Error: No se pudo inicializar el modelo Gemini."

    prompt = f"Eres Cali-Guía IA. Idioma de respuesta: {idioma_usuario}. Usuario pregunta: {query}\nLugares disponibles:\n"
    for s in contexto:
        prompt += f"- {s.get('nombre')} ({s.get('categoria')}) — {s.get('descripcion')[:200]}\n"
    prompt += "\nResponde en tono amable y sin inventar datos."

    try:
        response = llm_model.generate_content(prompt)
        return getattr(response, "text", str(response)).strip()
    except Exception as e:
        logging.error(f"Error al generar respuesta con Gemini: {e}")
        return "\n".join([f"- {s.get('nombre')}" for s in contexto])

# -------------------------------------------------------
# Función pública para ejecutar RAG
# -------------------------------------------------------
def ejecutar_rag(query: str, top_k: int = 5, usuario_info: dict = None, multilingual: bool = False):
    contexto = buscar_sitios_relevantes(query, top_k, multilingual=multilingual)
    respuesta = generar_respuesta_con_gemini(query, contexto, usuario_info) if GOOGLE_API_KEY else "Gemini no configurado"
    return {"contexto": contexto, "respuesta": respuesta}


# Nota: las funciones de construcción de índice (build_index_from_db) deben ser invocadas por:
# python scripts/precompute_embeddings.py
