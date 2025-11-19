# src/app_streamlit_chatbot_conversacional.py
import streamlit as st 
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import folium
from streamlit_folium import st_folium
import pandas as pd
import re

# Import internal modules (rag has lazy loading)
from db import usuarios_col, sitios_col
from filtrado_colaborativo import calcular_score_colaborativo
from filtrado_contenido import calcular_score_contenido
from filtrado_conocimiento import filtrar_por_usuario
from recomendador_hibrido import ranking_final
from clustering import cargar_datos_usuarios, entrenar_clustering, predecir_perfil
from rag import ejecutar_rag

# Config
st.set_page_config(
    page_title="🌴 Chatbot Planificador de Viajes Inteligente (Cali 🇨🇴)", 
    layout="wide"
)
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
ORS_API_KEY = os.getenv("ORS_API_KEY")

if not MONGO_URI or not ORS_API_KEY:
    st.error("❌ Define MONGO_URI y ORS_API_KEY en tu archivo .env")
    st.stop()

# Conexión a MongoDB y OpenRouteService (lazy client creation)
@st.cache_resource
def get_db_client(uri):
    return MongoClient(uri)

client = get_db_client(MONGO_URI)
db = client["turismo_cali"]

# Cargar modelo de embeddings una sola vez y en cache (ligero)
@st.cache_resource
def get_embedding_model():
    # Usar modelo pequeño y cacheado; si quieres otro, cambia MODEL_NAME en .env
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

embedding_model = get_embedding_model()

# Caching clustering model: evitamos re-entrenar cada petición
@st.cache_resource
def get_clustering_resources():
    df = cargar_datos_usuarios()
    # si df vacio, devolver placeholders
    if df is None or df.shape[0] == 0:
        return None, None, None
    kmeans_model, scaler, perfil_map = entrenar_clustering(df)
    return kmeans_model, scaler, perfil_map

kmeans_model, scaler, perfil_map = get_clustering_resources()

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🌴 Chatbot Planificador de Viajes Inteligente (Cali 🇨🇴)")
st.caption("Habla libremente con el asistente y recibe recomendaciones personalizadas 🤖")

# Show chat history
for msg in st.session_state.chat_history:
    role = "🧍‍♂️ Tú" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['content']}")

user_input = st.chat_input("Escribe tu mensaje aquí...")

def reiniciar_app():
    import streamlit.runtime.scriptrunner.script_runner as sr
    raise sr.RerunException(sr.RerunData())

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Procesando tu mensaje... 💭"):
        user_text = user_input.lower()
        # default profile
        presupuesto = 100
        tiempo_disponible = 8
        intereses = []
        presupuesto_match = re.search(r'presupuesto (\d+)', user_text)
        if presupuesto_match:
            presupuesto = float(presupuesto_match.group(1))
        tiempo_match = re.search(r'(\d+) días?', user_text)
        if tiempo_match:
            tiempo_disponible = float(tiempo_match.group(1))
        if "naturaleza" in user_text: intereses.append("naturaleza")
        if "ciudad" in user_text: intereses.append("ciudad")
        if "gastronomía" in user_text or "comida" in user_text: intereses.append("gastronomía")
        if "deporte" in user_text: intereses.append("deporte")
        if "museo" in user_text: intereses.append("museo")

        usuario = {
            "presupuesto": presupuesto,
            "tiempo_disponible": tiempo_disponible,
            "intereses": intereses or ["naturaleza", "ciudad"]
        }

        # CLUSTERING (si hay recursos)
        perfil = None
        cluster = None
        try:
            df_usuarios = cargar_datos_usuarios()
            if "intereses" not in df_usuarios.columns:
                df_usuarios["intereses"] = [[] for _ in range(len(df_usuarios))]
            df_usuarios = pd.concat([df_usuarios, pd.DataFrame([usuario])], ignore_index=True)
            if kmeans_model is not None:
                # solo predecir perfil usando modelo cacheado
                cluster, perfil = predecir_perfil(usuario, kmeans_model, scaler, perfil_map, [])
        except Exception:
            cluster, perfil = None, None

        # FILTRADO BASADO EN CONOCIMIENTO
        sitios = filtrar_por_usuario(usuario)
        if not sitios:
            sitios = list(sitios_col.find())

        # FILTRADOS
        score_colab = calcular_score_colaborativo(usuario, sitios, int(cluster) if cluster is not None else 0)
        score_contenido = calcular_score_contenido(usuario, embedding_model)
        sitios_ranked = ranking_final(sitios, score_colab, score_contenido)
        sitios_ranked = sitios_ranked[:300]  #if sitios_ranked else sitios[:5]

        # CONTEXTO PARA RAG (solo pasar info summary, RAG hará search en FAISS cached)
        sitios_contexto = " | ".join([
            f"{s.get('nombre_google', s.get('nombre', 'Sitio'))} ({s.get('categoria', '')})"
            for s in sitios_ranked
        ])

        query_text = f"Usuario: {user_input}. Contexto de sitios turísticos disponibles: {sitios_contexto}"

        # RECOMENDACIÓN RAG (ejecución remota/local; la función es rápida si FAISS está cargado)
        try:
            resultado_rag = ejecutar_rag(query_text, top_k=5)
            recomendacion_rag = resultado_rag.get("respuesta", "")
            if not recomendacion_rag.strip():
                raise Exception("Respuesta vacía.")
        except Exception:
            recomendacion_rag = (
                "⚠️ No se pudo generar una respuesta con el modelo. "
                "Se muestran recomendaciones basadas en filtros internos."
            )

        # Guardar resultado
        ORIGEN = [3.437, -76.529]
        st.session_state.resultado = {
            "perfil": perfil,
            "sitios_ranked": sitios_ranked,
            "recomendacion_rag": recomendacion_rag,
            "origen": ORIGEN
        }

        bot_message = (
            "¡Listo! He generado recomendaciones basadas en tu consulta ✅\n\n"
            + recomendacion_rag
        )
        st.session_state.chat_history.append({"role": "bot", "content": bot_message})

        reiniciar_app()

# Mostrar mapa y recomendaciones si existen
if "resultado" in st.session_state:
    res = st.session_state.resultado
    m = folium.Map(location=res["origen"], zoom_start=12)
    folium.Marker(location=res["origen"], tooltip="Tu ubicación", icon=folium.Icon(color="blue")).add_to(m)
    for s in res["sitios_ranked"]:
        coord = [s.get("latitud", 3.437), s.get("longitud", -76.529)]
        folium.Marker(location=coord, tooltip=s.get("nombre_google", s.get("nombre", "Sitio"))).add_to(m)
    st_folium(m, width=700, height=400)

    st.markdown("### 🌟 Recomendaciones finales:")
    for i, s in enumerate(res["sitios_ranked"], start=1):
        st.markdown(f"**{i}. {s.get('nombre_google', s.get('nombre', 'Sitio'))}** — "
                    f"{s.get('categorías_google', s.get('categoria', ''))} — "
                    f"Score final: {s.get('score_final', 0):.2f}")

    st.markdown("### 🤖 Recomendación personalizada:")
    st.markdown(res["recomendacion_rag"])

if st.button("🧹 Limpiar conversación"):
    st.session_state.chat_history = []
    if "resultado" in st.session_state:
        del st.session_state["resultado"]
    reiniciar_app()