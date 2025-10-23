# ========================================================
# RAG CON MONGO + GEMINI 2.5-FLASH (Versión Mejorada con Contexto Global)
# ========================================================

import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
import torch
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# ========================================================
# [1] Configuración Inicial
# ========================================================

# --- Logging ---
log_file = 'rag_gemini.log'
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info("Inicio del módulo RAG con Gemini 2.5-FLASH mejorado.")

# --- Cargar variables de entorno ---
dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path=dotenv_path)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "turismo_cali")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sitios")
MODEL_NAME = os.getenv("MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Modelo Gemini ---
GEMINI_MODEL = "models/gemini-2.5-flash"

# ========================================================
# [2] Conexión a MongoDB
# ========================================================
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    places_col = db[COLLECTION_NAME]
    logging.info(f"Conectado a MongoDB: Base={DB_NAME}, Colección={COLLECTION_NAME}")
except Exception as e:
    logging.critical(f"No se pudo conectar a MongoDB: {e}")
    raise SystemExit(e)

# ========================================================
# [3] Cargar modelo de embeddings
# ========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Dispositivo seleccionado: {DEVICE}")

model = SentenceTransformer(MODEL_NAME, device=DEVICE)
logging.info("Modelo de embeddings cargado exitosamente.")

# ========================================================
# [4] Cargar corpus y generar embeddings con contexto ampliado
# ========================================================
corpus_sitios = list(places_col.find({}))
logging.info(f"Se cargaron {len(corpus_sitios)} registros desde MongoDB.")

corpus_texts = [
    f"{s.get('nombre', '')}. {s.get('categoria', '')}. {s.get('descripcion', '')}. "
    f"Ubicado en {s.get('direccion', '')}. Comentarios: {s.get('comentarios', '')}"
    for s in corpus_sitios
]

corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, device=DEVICE)
logging.info(f"Embeddings generados para {len(corpus_embeddings)} sitios con contexto extendido.")

# ========================================================
# [5] Búsqueda (Retrieve)
# ========================================================
def buscar_sitios_relevantes(query: str, top_k: int = 5):
    """
    Busca los sitios más relevantes en la base de datos según la consulta.
    Retorna una lista de sitios con información completa.
    """
    query_embedding = model.encode(query, convert_to_tensor=True, device=DEVICE)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores / torch.norm(cos_scores)  # Normalización ligera

    top_results = torch.topk(cos_scores, k=min(top_k, len(corpus_sitios)))

    resultados = []
    for score, idx in zip(top_results[0], top_results[1]):
        sitio = corpus_sitios[idx]
        resultados.append({
            "score": round(score.item(), 4),
            "nombre": sitio.get("nombre", "No disponible"),
            "direccion": sitio.get("direccion", "No disponible"),
            "categoria": sitio.get("categoria", "No disponible"),
            "descripcion": sitio.get("descripcion", "No disponible"),
            "latitud": sitio.get("latitud", None),
            "longitud": sitio.get("longitud", None),
            "comentarios": sitio.get("comentarios", "No hay comentarios")
        })
    return resultados

# ========================================================
# [6] Configurar Gemini 2.5-FLASH
# ========================================================
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logging.info("API de Google Gemini configurada exitosamente.")
        logging.info(f"Modelo seleccionado: {GEMINI_MODEL}")
    except Exception as e:
        logging.critical(f"Error al configurar la API de Gemini: {e}")
else:
    logging.warning("No se encontró la variable GOOGLE_API_KEY. Gemini no estará disponible.")

# ========================================================
# [7] Generación de respuesta con comprensión contextual
# ========================================================
def generar_respuesta_con_gemini(query: str, contexto: list, usuario_info: dict = None):
    """
    Genera respuesta usando Gemini 2.5-FLASH basada en la consulta,
    contexto y datos del usuario. Comprende intención y contexto global.
    """
    if not GOOGLE_API_KEY:
        return "Error: Falta la clave API de Google. No se puede generar la respuesta."

    if not contexto:
        return "No se encontraron lugares relevantes para tu consulta."

    try:
        llm_model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        logging.error(f"No se pudo inicializar el modelo Gemini '{GEMINI_MODEL}': {e}")
        return "Error: No se pudo inicializar el modelo Gemini."

    # Construcción del prompt con comprensión de intención
    prompt = f"""
Eres **Cali-Guía IA**, un asistente turístico inteligente y empático que conoce a fondo Cali, Colombia.
Tu misión es entender la intención completa del usuario, más allá de palabras clave, y recomendar lugares adecuados.

🧭 Pregunta o contexto del turista:
"{query}"

{f"💰 Presupuesto: {usuario_info.get('presupuesto', 'N/A')}" if usuario_info else ""}
{f"⏱️ Tiempo disponible: {usuario_info.get('tiempo_disponible', 'N/A')} días" if usuario_info else ""}
{f"🎯 Intereses: {', '.join(usuario_info.get('intereses', []))}" if usuario_info else ""}

📍 Información de lugares relevantes:
"""
    for i, sitio in enumerate(contexto):
        prompt += f"""
--- Lugar {i+1} ---
Nombre: {sitio['nombre']}
Categoría: {sitio['categoria']}
Descripción: {sitio['descripcion']}
Dirección: {sitio['direccion']}
Lat/Lng: {sitio['latitud']}, {sitio['longitud']}
Opiniones: {sitio['comentarios'][:400]}...
"""

    prompt += """
🎯 Instrucciones para la respuesta:
- Analiza la intención del usuario (qué busca, qué tipo de experiencia quiere).
- Usa SOLO los datos de los lugares proporcionados.
- Justifica por qué cada lugar se adapta al usuario.
- Si el usuario pregunta algo general (ej. historia, clima, rutas), responde brevemente con contexto real de Cali.
- Mantén un tono amigable, natural y motivador.
- No inventes información ni menciones lugares no listados.
"""

    try:
        response = llm_model.generate_content(prompt)
        return getattr(response, "text", str(response)).strip()
    except Exception as e:
        logging.error(f"Error al generar respuesta con Gemini: {e}")
        fallback_text = "No se pudo generar una respuesta precisa de Gemini.\nAquí están los lugares más relevantes:\n"
        for sitio in contexto:
            fallback_text += f"- {sitio.get('nombre', 'Sitio desconocido')}\n"
        return fallback_text

# ========================================================
# [8] Ejecución del flujo completo RAG
# ========================================================
def ejecutar_rag(query: str, top_k: int = 3, usuario_info: dict = None):
    """
    Ejecuta el flujo completo del RAG:
    1. Recupera contexto desde MongoDB
    2. Genera respuesta con Gemini 2.5-FLASH
    """
    contexto = buscar_sitios_relevantes(query, top_k)
    respuesta = generar_respuesta_con_gemini(query, contexto, usuario_info)
    return {"contexto": contexto, "respuesta": respuesta}
