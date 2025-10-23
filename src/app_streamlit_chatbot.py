import streamlit as st 
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import folium
from streamlit_folium import st_folium
import openrouteservice
import pandas as pd

# --- Importar módulos internos ---
from db import usuarios_col, sitios_col
from filtrado_colaborativo import calcular_score_colaborativo
from filtrado_contenido import calcular_score_contenido
from filtrado_conocimiento import filtrar_por_usuario
from recomendador_hibrido import ranking_final
from clustering import cargar_datos_usuarios, entrenar_clustering, predecir_perfil
from rag import generar_recomendacion

# === CONFIGURACIÓN ===
st.set_page_config(page_title="Planificador de Viajes Inteligente", layout="wide")
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

# --- Historial del chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Interfaz principal ---
st.title("🌴 Chatbot Planificador de Viajes Inteligente (Cali 🇨🇴)")

# --- Mostrar historial del chat ---
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**Tú:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# --- Si ya hay resultados guardados, mostrarlos debajo del chat ---
if "resultado" in st.session_state:
    res = st.session_state["resultado"]
    st.markdown(f"### ✅ Perfil asignado: {res['perfil']}")

    # --- Mapa de recomendaciones ---
    m = folium.Map(location=res["origen"], zoom_start=12)
    folium.Marker(location=res["origen"], tooltip="Tu ubicación", icon=folium.Icon(color="blue")).add_to(m)
    for s in res["sitios_ranked"]:
        coord = [s.get("latitud", 3.437), s.get("longitud", -76.529)]
        folium.Marker(
            location=coord,
            tooltip=s.get("nombre_google", s.get("nombre", "Sitio"))
        ).add_to(m)
    st_folium(m, width=700, height=500)

    # --- Lista de recomendaciones ---
    st.markdown("### 🌟 Recomendaciones finales:")
    for i, s in enumerate(res["sitios_ranked"], start=1):
        st.markdown(
            f"**{i}. {s.get('nombre_google', s.get('nombre', 'Sitio'))}** — "
            f"{s.get('categorías_google', s.get('categoria', ''))} — "
            f"Score final: {s.get('score_final', 0):.2f}"
        )

    # --- Texto generado por RAG ---
    st.markdown("### 🤖 Recomendación personalizada:")
    st.markdown(res["recomendacion_rag"])

# --- Preguntas del chatbot paso a paso ---
preguntas = [
    "¿Qué parte de Cali quieres conocer?",
    "Presupuesto estimado",
    "Número de días de viaje",
    "¿Con quién viajas?",
    "¿Prefieres naturaleza o ciudad?",
    "Preferencias gastronómicas",
    "Actividades favoritas",
    "Facilidad de acceso (Sí/No)",
    "Tipo de alojamiento preferido"
]

indice_actual = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])

# --- Mostrar preguntas una por una ---
if indice_actual < len(preguntas):
    with st.form(key="chat_form"):
        respuesta = st.text_input(f"💬 {preguntas[indice_actual]}")
        enviar = st.form_submit_button("Enviar")

        if enviar and respuesta.strip():
            st.session_state.chat_history.append({"role": "user", "content": respuesta})
            st.rerun()

else:
    # --- Todos los inputs recogidos ---
    user_inputs = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]

    # --- Coordenadas del usuario ---
    user_lat = st.number_input("Latitud", value=3.437, format="%.6f")
    user_lon = st.number_input("Longitud", value=-76.529, format="%.6f")
    ORIGEN = [user_lat, user_lon]

    # --- Diccionario del usuario ---
    usuario = {
        "presupuesto": float(user_inputs[1]) if user_inputs[1].replace(".", "", 1).isdigit() else 100,
        "tiempo_disponible": float(user_inputs[2]) if user_inputs[2].replace(".", "", 1).isdigit() else 8,
        "intereses": user_inputs[4:7]  # naturaleza / gastronomía / actividades
    }

    # --- Botón para generar recomendaciones ---
    if st.button("🚀 Generar recomendaciones"):
        with st.spinner("Generando recomendaciones..."):

            # === CLUSTERING ===
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

            # === FILTRADO BASADO EN CONOCIMIENTO ===
            sitios = filtrar_por_usuario(usuario)
            if not sitios:
                sitios = list(sitios_col.find().limit(5))
                st.warning("No se encontraron sitios exactos para tus preferencias. Mostrando recomendaciones generales.")

            # === MODELO DE EMBEDDINGS ===
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

            # === FILTRADOS ===
            score_colab = calcular_score_colaborativo(usuario, sitios, int(cluster))
            score_contenido = calcular_score_contenido(usuario, model)
            sitios_ranked = ranking_final(sitios, score_colab, score_contenido)

            if not sitios_ranked:
                sitios_ranked = sitios[:5]

            sitios_ranked = sitios_ranked[:5]

            # === RECOMENDACIÓN RAG ===
            query_text = " ".join(usuario["intereses"])
            recomendacion_rag = generar_recomendacion(query_text, sitios_ranked)

            # === GUARDAR RESULTADOS EN SESIÓN ===
            st.session_state.resultado = {
                "perfil": perfil,
                "sitios_ranked": sitios_ranked,
                "recomendacion_rag": recomendacion_rag,
                "origen": ORIGEN
            }

            st.session_state.chat_history.append(
                {"role": "bot", "content": "Recomendaciones generadas con éxito ✅"}
            )

            st.rerun()
