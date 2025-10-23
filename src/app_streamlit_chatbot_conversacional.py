import streamlit as st 
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import folium
from streamlit_folium import st_folium
import pandas as pd
import openrouteservice
import re

# --- Importar módulos internos ---
from db import usuarios_col, sitios_col
from filtrado_colaborativo import calcular_score_colaborativo
from filtrado_contenido import calcular_score_contenido
from filtrado_conocimiento import filtrar_por_usuario
from recomendador_hibrido import ranking_final
from clustering import cargar_datos_usuarios, entrenar_clustering, predecir_perfil
from rag import ejecutar_rag

# === CONFIGURACIÓN ===
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

# --- Conexión a MongoDB y OpenRouteService ---
client = MongoClient(MONGO_URI)
db = client["turismo_cali"]
ors_client = openrouteservice.Client(key=ORS_API_KEY)

# --- Cargar modelo de embeddings una vez ---
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# --- Inicializar historial de chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Interfaz principal ---
st.title("🌴 Chatbot Planificador de Viajes Inteligente (Cali 🇨🇴)")
st.caption("Habla libremente con el asistente y recibe recomendaciones personalizadas 🤖")

# --- Mostrar historial del chat ---
for msg in st.session_state.chat_history:
    role = "🧍‍♂️ Tú" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['content']}")

# --- Caja de entrada conversacional ---
user_input = st.chat_input("Escribe tu mensaje aquí...")

def reiniciar_app():
    """Reinicia la app de manera compatible con la versión actual de Streamlit"""
    import streamlit.runtime.scriptrunner.script_runner as sr
    raise sr.RerunException(sr.RerunData())

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Procesando tu mensaje... 💭"):
        # --- Interpretación básica del texto ---
        user_text = user_input.lower()

        # Valores por defecto
        presupuesto = 100
        tiempo_disponible = 8
        intereses = []

        # Extracción simple de datos
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

        # --- CLUSTERING ---
        df_usuarios = cargar_datos_usuarios()
        if "intereses" not in df_usuarios.columns:
            df_usuarios["intereses"] = [[] for _ in range(len(df_usuarios))]
        df_usuarios = pd.concat([df_usuarios, pd.DataFrame([usuario])], ignore_index=True)

        kmeans_model, scaler, perfil_map = entrenar_clustering(df_usuarios)
        df_intereses = df_usuarios['intereses'].apply(
            lambda x: '|'.join(x) if isinstance(x, list) else ""
        ).str.get_dummies()
        columnas_intereses = list(df_intereses.columns)
        cluster, perfil = predecir_perfil(usuario, kmeans_model, scaler, perfil_map, columnas_intereses)

        # --- FILTRADO BASADO EN CONOCIMIENTO ---
        sitios = filtrar_por_usuario(usuario)
        if not sitios:
            sitios = list(sitios_col.find().limit(5))

        # --- FILTRADOS ---
        score_colab = calcular_score_colaborativo(usuario, sitios, int(cluster))
        score_contenido = calcular_score_contenido(usuario, st.session_state.embedding_model)
        sitios_ranked = ranking_final(sitios, score_colab, score_contenido)
        sitios_ranked = sitios_ranked[:5] if sitios_ranked else sitios[:5]

        # --- CONTEXTO PARA EL RAG ---
        sitios_contexto = " | ".join([
            f"{s.get('nombre_google', s.get('nombre', 'Sitio'))} ({s.get('categoria', '')})"
            for s in sitios_ranked
        ])

        # Ahora el modelo recibe el mensaje original + contexto de sitios
        query_text = f"Usuario: {user_input}. Contexto de sitios turísticos disponibles: {sitios_contexto}"

        # --- RECOMENDACIÓN RAG con fallback ---
        try:
            resultado_rag = ejecutar_rag(query_text, top_k=5)
            recomendacion_rag = resultado_rag.get("respuesta", "")
            if "exceeded" in recomendacion_rag.lower() or not recomendacion_rag.strip():
                raise Exception("Respuesta vacía o límite de cuota.")
        except Exception:
            recomendacion_rag = (
                "⚠️ No se pudo generar una respuesta con el modelo. "
                "Se muestran recomendaciones basadas en filtros internos."
            )

        # --- Guardar resultado en sesión ---
        ORIGEN = [3.437, -76.529]  # Coordenadas por defecto
        st.session_state.resultado = {
            "perfil": perfil,
            "sitios_ranked": sitios_ranked,
            "recomendacion_rag": recomendacion_rag,
            "origen": ORIGEN
        }

        # --- Guardar respuesta del bot ---
        bot_message = (
            "¡Listo! He generado recomendaciones basadas en tu consulta ✅\n\n"
            + recomendacion_rag
        )
        st.session_state.chat_history.append({"role": "bot", "content": bot_message})

        reiniciar_app()

# --- Mostrar mapa y recomendaciones finales si existen ---
if "resultado" in st.session_state:
    res = st.session_state.resultado

    # Mapa
    m = folium.Map(location=res["origen"], zoom_start=12)
    folium.Marker(location=res["origen"], tooltip="Tu ubicación", icon=folium.Icon(color="blue")).add_to(m)
    for s in res["sitios_ranked"]:
        coord = [s.get("latitud", 3.437), s.get("longitud", -76.529)]
        folium.Marker(location=coord, tooltip=s.get("nombre_google", s.get("nombre", "Sitio"))).add_to(m)
    st_folium(m, width=700, height=400)

    # Lista de recomendaciones
    st.markdown("### 🌟 Recomendaciones finales:")
    for i, s in enumerate(res["sitios_ranked"], start=1):
        st.markdown(f"**{i}. {s.get('nombre_google', s.get('nombre', 'Sitio'))}** — "
                    f"{s.get('categorías_google', s.get('categoria', ''))} — "
                    f"Score final: {s.get('score_final', 0):.2f}")

    # Texto generado por RAG
    st.markdown("### 🤖 Recomendación personalizada:")
    st.markdown(res["recomendacion_rag"])

# --- Botón para limpiar chat ---
if st.button("🧹 Limpiar conversación"):
    st.session_state.chat_history = []
    if "resultado" in st.session_state:
        del st.session_state["resultado"]
    reiniciar_app()
