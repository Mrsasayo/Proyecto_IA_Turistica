# src/app_streamlit_chatbot_conversacional_mejorado.py
import streamlit as st 
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import folium
from streamlit_folium import st_folium
import pandas as pd
import re
from datetime import datetime
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # Para resultados consistentes en la detección de idioma
from googletrans import Translator
translator = Translator()

# -------------------- Función de traducción --------------------
def traducir_texto(texto, destino="es"):
    try:
        return translator.translate(texto, dest=destino).text
    except Exception:
        return texto

# -------------------- Nueva función traducir_ui --------------------
def traducir_ui(texto, destino=None):
    if destino is None:
        destino = st.session_state['idioma_usuario']
    try:
        # Traducir solo si el idioma no es español
        if destino == "es":
            return texto
        return translator.translate(texto, dest=destino).text
    except Exception:
        return texto

# Import internal modules (rag has lazy loading)
from db import usuarios_col, sitios_col
from filtrado_colaborativo import calcular_score_colaborativo
from filtrado_contenido import calcular_score_contenido
from filtrado_conocimiento import filtrar_por_usuario
from recomendador_hibrido import ranking_final
from clustering import cargar_datos_usuarios, entrenar_clustering, predecir_perfil
from rag import ejecutar_rag
from evaluador_modelo_pdf import evaluar_rag_completo_v2

# -------------------- CONFIG -------------------------------------------------
st.set_page_config(
    page_title="🌴 Chatbot Planificador de Viajes Inteligente (Cali 🇨🇴)", 
    layout="wide",
    initial_sidebar_state="expanded"
)
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
ORS_API_KEY = os.getenv("ORS_API_KEY")

if not MONGO_URI or not ORS_API_KEY:
    st.error("❌ Define MONGO_URI y ORS_API_KEY en tu archivo .env")
    st.stop()

# -------------------- DB / MODEL (cached) ----------------------------------
@st.cache_resource
def get_db_client(uri):
    return MongoClient(uri)

client = get_db_client(MONGO_URI)
db = client["turismo_cali"]

@st.cache_resource
def get_embedding_model():
    from rag import get_embedding_model as rag_get_embedding_model
    return rag_get_embedding_model(multilingual=True)

embedding_model = get_embedding_model()

@st.cache_resource
def get_clustering_resources():
    df = cargar_datos_usuarios()
    if df is None or df.shape[0] == 0:
        return None, None, None
    kmeans_model, scaler, perfil_map = entrenar_clustering(df)
    return kmeans_model, scaler, perfil_map

kmeans_model, scaler, perfil_map = get_clustering_resources()

# -------------------- SESSION STATE init -----------------------------------
default_state = {
    "chat_sessions": [{"name": "Chat 1", "history": [], "resultado": None}],
    "active_chat": 0,
    "dark_mode": False,
    "listas": {},
    "resultado": None,
    "editing": None,
    "idioma_usuario": "es"
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------------- THEME OVERRIDE (global) -------------------------------
def apply_theme(dark: bool):
    if dark:
        primary_bg, panel_bg, text_color = "#D9D9D9", "#E6E6E6", "#1A1A1A"
        bubble_user, bubble_bot, accent = "#B3B3B3", "#999999", "#4D4D4D"
    else:
        primary_bg, panel_bg, text_color = "#F6F7F8", "#FFFFFF", "#0b1220"
        bubble_user, bubble_bot, accent = "#DCF8C6", "#F1F0F0", "#06b6d4"

    css = f"""
    <style>
    :root {{
      --primary-bg: {primary_bg};
      --panel-bg: {panel_bg};
      --text-color: {text_color};
      --bubble-user: {bubble_user};
      --bubble-bot: {bubble_bot};
      --accent: {accent};
    }}
    .stApp {{ background: var(--primary-bg); color: var(--text-color); }}
    .main .block-container {{ background: transparent; }}
    .css-1d391kg .block-container, .css-1oe6wy1 .block-container {{ background: var(--panel-bg); }}
    .sidebar .sidebar-content {{ background: var(--panel-bg); }}
    .chat-wrap {{ padding: 14px; }}
    .bubble {{ display:inline-block; padding:12px 14px; border-radius:18px; max-width:72%; word-wrap:break-word; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
    .bubble.user {{ background: var(--bubble-user); color: var(--text-color); margin-left: auto; border-bottom-right-radius: 4px; }}
    .bubble.bot  {{ background: var(--bubble-bot); color: var(--text-color); margin-right: auto; border-bottom-left-radius: 4px; }}
    .msg-row {{ display:flex; gap:12px; align-items:flex-end; margin-bottom:8px; }}
    .msg-row.user {{ justify-content: flex-end; }}
    .msg-row.bot {{ justify-content: flex-start; }}
    .avatar {{ width:36px; height:36px; border-radius:50%; }}
    .meta {{ font-size:11px; opacity:0.7; margin-top:6px; }}
    .dots {{ display:inline-block; }}
    .dot {{ height:6px; width:6px; margin:0 2px; background:rgba(0,0,0,0.2); border-radius:50%; display:inline-block; animation: blink 1.2s infinite; }}
    .dot:nth-child(2) {{ animation-delay:0.2s; }}
    .dot:nth-child(3) {{ animation-delay:0.4s; }}
    @keyframes blink {{ 0% {{ opacity:0.2; transform: translateY(0); }} 50% {{ opacity:1; transform: translateY(-4px); }} 100% {{ opacity:0.2; transform: translateY(0); }} }}
    .stExpander{{ border-radius:8px; padding:6px; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_theme(st.session_state.dark_mode)

# -------------------- SIDEBAR (CONFIGURACIÓN) -------------------------------
with st.sidebar:
    st.title(traducir_ui("⚙️ Configuración"))
    with st.expander(traducir_ui("Opciones"), expanded=True):
        # Tema
        new_theme = st.selectbox(traducir_ui("Tema (claro/oscuro)"), options=[traducir_ui("Oscuro"), traducir_ui("Claro")], index=0 if st.session_state.dark_mode else 1)
        chosen_dark = new_theme == traducir_ui("Oscuro")
        if chosen_dark != st.session_state.dark_mode:
            st.session_state.dark_mode = chosen_dark
            apply_theme(st.session_state.dark_mode)
            st.experimental_rerun()

        # Nuevo chat
        st.markdown("---")
        if st.button(traducir_ui("🆕 Nuevo chat")):
            name = f"{traducir_ui('Chat')} {len(st.session_state.chat_sessions)+1}"
            st.session_state.chat_sessions.append({"name": name, "history": [], "resultado": None})
            st.session_state.active_chat = len(st.session_state.chat_sessions)-1
            st.experimental_rerun()

        # Historial de chats
        st.write(traducir_ui("### Historial de chats"))
        names = [c["name"] for c in st.session_state.chat_sessions]
        sel = st.selectbox(traducir_ui("Abrir chat guardado"), options=names, index=st.session_state.active_chat)
        if sel and names.index(sel) != st.session_state.active_chat:
            st.session_state.active_chat = names.index(sel)
            st.experimental_rerun()

        # Descargar itinerario
        st.markdown("---")   
        if st.button(traducir_ui("📥 Descargar último itinerario")):
            last = st.session_state.chat_sessions[st.session_state.active_chat].get("resultado")
            if last and last.get("recomendacion_rag"):
                payload = last.get("recomendacion_rag")
                st.download_button(
                    traducir_ui("⬇️ Descargar (TXT)"),
                    payload.encode("utf-8"),
                    file_name="itinerario_trevly.txt"
                )
            else:
                st.info(traducir_ui("No hay itinerario generado aún."))

        # Limpiar conversación
        st.markdown("---")
        if st.button(traducir_ui("🧹 Limpiar conversación")):
            st.session_state.chat_sessions[st.session_state.active_chat]["history"] = []
            st.session_state.chat_sessions[st.session_state.active_chat]["resultado"] = None
            st.session_state.resultado = None
            st.experimental_rerun()

# -------------------- Helpers: procesar mensaje del usuario ------------------
def detectar_idioma(texto):
    try:
        return detect(texto)
    except Exception:
        return "desconocido"
    
def procesar_mensaje_usuario(user_text):
    """
    Procesa el mensaje del usuario:
    - Detecta idioma
    - Extrae presupuesto, tiempo e intereses
    - Filtra y puntúa sitios turísticos
    - Genera recomendación con RAG
    - Evalúa automáticamente la respuesta del RAG usando los sitios rankeados
    """ 
    idioma = detectar_idioma(user_text)
    presupuesto, tiempo_disponible, intereses = 100, 8, []
    user_text_low = user_text.lower()
    presupuesto_match = re.search(r'presupuesto (\d+)', user_text_low)
    if presupuesto_match:
        presupuesto = float(presupuesto_match.group(1))
    tiempo_match = re.search(r'(\d+) días?', user_text_low)
    if tiempo_match:
        tiempo_disponible = float(tiempo_match.group(1))
    for keyword, tag in [("naturaleza","naturaleza"),("ciudad","ciudad"),("gastronomía","gastronomía"),("comida","gastronomía"),("deporte","deporte"),("museo","museo")]:
        if keyword in user_text_low:
            intereses.append(tag)
    usuario = {"presupuesto": presupuesto, "tiempo_disponible": tiempo_disponible, "intereses": intereses or ["naturaleza","ciudad"]}

    cluster, perfil = None, None
    try:
        df_usuarios = cargar_datos_usuarios()
        if "intereses" not in df_usuarios.columns:
            df_usuarios["intereses"] = [[] for _ in range(len(df_usuarios))]
        df_usuarios = pd.concat([df_usuarios, pd.DataFrame([usuario])], ignore_index=True)
        if kmeans_model is not None:
            cluster, perfil = predecir_perfil(usuario, kmeans_model, scaler, perfil_map, [])
    except Exception:
        cluster, perfil = None, None

    sitios = filtrar_por_usuario(usuario)
    if not sitios:
        sitios = list(sitios_col.find())
    score_colab = calcular_score_colaborativo(usuario, sitios, int(cluster) if cluster is not None else 0)
    score_contenido = calcular_score_contenido(usuario, embedding_model)
    sitios_ranked = ranking_final(sitios, score_colab, score_contenido)[:5]
    contexto = " | ".join([s.get("nombre_google", s.get("nombre")) for s in sitios_ranked])
    query_text = f"Usuario: {user_text}. Contexto: {contexto}"

    try:
        resultado_rag = ejecutar_rag(query_text, top_k=5)
        recomendacion_rag = resultado_rag.get("respuesta", "")

        # ---------------- EVALUACIÓN AUTOMÁTICA DEL MODELO ----------------
        resultados_recuperados = [s.get("nombre_google", s.get("nombre")) for s in sitios_ranked]
        # Se asume que todos los sitios rankeados son "relevantes"
        documentos_relevantes = resultados_recuperados.copy()
        # Se usa la misma recomendación como "respuesta_modelo"
        metrics = evaluar_rag_completo_v2(
            query=query_text,
            resultados_recuperados=resultados_recuperados,
            documentos_relevantes=documentos_relevantes,
            respuesta_modelo=recomendacion_rag,
            respuesta_esperada=contexto,  # contexto generado a partir de los sitios
            k=len(resultados_recuperados)
        )
    except Exception:
        recomendacion_rag = "⚠️ No se pudo generar la respuesta del modelo."
        metrics = None

    ORIGEN = [3.437, -76.529]
    resultado = {
        "perfil": perfil,
        "sitios_ranked": sitios_ranked,
        "recomendacion_rag": recomendacion_rag,
        "origen": ORIGEN,
        "idioma_usuario": idioma,
        "metrics": metrics
    }
    bot_resp = f"¡Listo! He generado recomendaciones basadas en tu consulta ✅ (Idioma detectado: {idioma})"
    return bot_resp, resultado

# -------------------- MAIN LAYOUT (LEFT: sessions, RIGHT: chat) -------------
col_left, col_right = st.columns([1,3], gap="small")

with col_left:
    st.markdown(traducir_ui("### 💬 Chats"))
    for i, chat in enumerate(st.session_state.chat_sessions):
        c1, c2 = st.columns([3,1])
        with c1:
            # Generar nombre traducido dinámicamente
            chat_nombre = traducir_ui(f"Chat {i+1}")
            if st.button(f"{i+1}. {chat_nombre}", key=f"left_chat_{i}"):
                st.session_state.active_chat = i
                st.experimental_rerun()
        with c2:
            if st.button("🗑️", key=f"del_chat_{i}"):
                if len(st.session_state.chat_sessions) == 1:
                    st.session_state.chat_sessions = [{"name": traducir_ui("Chat 1"), "history": [], "resultado": None}]
                    st.session_state.active_chat = 0
                else:
                    del st.session_state.chat_sessions[i]
                    if st.session_state.active_chat >= len(st.session_state.chat_sessions):
                        st.session_state.active_chat = max(0, len(st.session_state.chat_sessions)-1)
                st.experimental_rerun()
    st.markdown("---")
    st.caption(traducir_ui("Usa el panel 'Configuración' para crear o limpiar chats. Elimina con la papelera."))

# ---------- DERECHA: selector de idioma compacto ----------
with col_right:
    cols = st.columns([6,1])
    with cols[1]:
        idioma_usuario = st.selectbox(
            traducir_ui("Idioma"),
            options=["es", "en", "fr", "de", "pt", "it", "zh", "ja", "ru", "ar"],
            index=["es", "en", "fr", "de", "pt", "it", "zh", "ja", "ru", "ar"].index(st.session_state['idioma_usuario']),
            label_visibility="collapsed"
        )
        if idioma_usuario != st.session_state['idioma_usuario']:
            st.session_state['idioma_usuario'] = idioma_usuario
            for msg in st.session_state.chat_sessions[st.session_state.active_chat]['history']:
                if msg['role'] == 'bot':
                    msg['content'] = traducir_texto(msg['content'], destino=idioma_usuario)
            st.experimental_rerun()


# -------------------- Chat principal --------------------
with col_right:
    # Obtener el nombre dinámico del chat según idioma
    chat_id = st.session_state.chat_sessions[st.session_state.active_chat].get("id", f"chat_{st.session_state.active_chat+1}")
    chat_nombre = traducir_ui(f"Chat {st.session_state.active_chat+1}")  # traducido
    st.markdown(f"## 🌴 {chat_nombre}")

    st.markdown(traducir_ui("Interactúa con Trevly — escribe en la caja y presiona Enter"))

    st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)
    history = st.session_state.chat_sessions[st.session_state.active_chat]["history"]
    resultado_chat = st.session_state.chat_sessions[st.session_state.active_chat].get("resultado", None)

    BOT_AVATAR = "https://img.icons8.com/fluency/48/000000/robot.png"
    USER_AVATAR = "https://img.icons8.com/fluency/48/000000/user-male-circle.png"

    for idx, msg in enumerate(history):
        role = msg.get("role")
        ts = msg.get("ts", "")
        content = msg.get("content", "")
        idioma_msg = msg.get("idioma", "")
        if role == "bot":
            st.markdown(f"""
            <div class='msg-row bot'>
                <img src='{BOT_AVATAR}' class='avatar'/>
                <div>
                    <div class='bubble bot'>{content}</div>
                    <div class='meta'>🤖 Trevly — {ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            cols = st.columns([1,8,1])
            with cols[0]:
                st.image(USER_AVATAR, width=36)
            with cols[1]:
                st.markdown(f"<div class='bubble user'>{content}</div>", unsafe_allow_html=True)
                if idioma_msg:
                    st.markdown(f"<div class='meta'>🌐 {traducir_ui('Idioma detectado')}: {idioma_msg}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='meta'>👤 {traducir_ui('Tú')} — {ts}</div>", unsafe_allow_html=True)
            with cols[2]:
                if st.session_state.editing != idx:
                    if st.button(traducir_ui("Editar"), key=f"edit_btn_{st.session_state.active_chat}_{idx}"):
                        st.session_state.editing = idx
                        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Edición de mensajes --------------------
if st.session_state.editing is not None:
    edit_idx = st.session_state.editing
    if 0 <= edit_idx < len(history) and history[edit_idx]["role"] == "user":
        st.markdown("---")
        st.markdown(traducir_ui("### ✏️ Editar tu mensaje"))
        new_text = st.text_area(
            traducir_ui("Nuevo texto:"), 
            value=history[edit_idx]["content"], 
            key=f"edit_area_{st.session_state.active_chat}_{edit_idx}", 
            height=120
        )
        col_save, col_cancel = st.columns([1,1])
        with col_save:
            if st.button(traducir_ui("Guardar edición"), key=f"save_edit_{st.session_state.active_chat}_{edit_idx}"):
                history[edit_idx]["content"] = new_text
                history[edit_idx]["ts"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                placeholder = st.empty()
                placeholder.markdown(
                    f"<div class='msg-row bot'><img src='{BOT_AVATAR}' class='avatar'/><div><div class='bubble bot'><span class='dots'><span class='dot'></span><span class='dot'></span><span class='dot'></span></span></div><div class='meta'>🤖 {traducir_ui('Trevly — escribiendo...')}</div></div></div>",
                    unsafe_allow_html=True
                )

                with st.spinner(traducir_ui("Procesando tu mensaje editado... 💭")):
                    bot_resp, nuevo_resultado = procesar_mensaje_usuario(new_text)
                    bot_resp = traducir_ui(bot_resp)

                st.session_state.chat_sessions[st.session_state.active_chat]["resultado"] = nuevo_resultado
                st.session_state.resultado = nuevo_resultado

                bot_msg_idx = None
                for j in range(edit_idx+1, len(history)):
                    if history[j]["role"] == "bot":
                        bot_msg_idx = j
                        break

                if bot_msg_idx is not None:
                    history[bot_msg_idx]["content"] = bot_resp
                    history[bot_msg_idx]["ts"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                else:
                    history.append({"role":"bot","content":bot_resp,"ts":datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

                placeholder.empty()
                st.session_state.editing = None
                st.experimental_rerun()
        with col_cancel:
            if st.button(traducir_ui("Cancelar"), key=f"cancel_edit_{st.session_state.active_chat}_{edit_idx}"):
                st.session_state.editing = None
                st.experimental_rerun()


# -------------------- Input Form --------------------
with st.form(key=f'chat_form_{st.session_state.active_chat}', clear_on_submit=True):
    user_input = st.text_input(traducir_ui("Escribe tu mensaje aquí..."), key=f"input_{st.session_state.active_chat}")
    submitted = st.form_submit_button(traducir_ui("Enviar"))
    if submitted and user_input:
        idioma = detectar_idioma(user_input)
        history.append({
            "role":"user",
            "content":user_input,
            "ts":datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "idioma": idioma
        })

        placeholder = st.empty()
        placeholder.markdown(
            f"<div class='msg-row bot'><img src='{BOT_AVATAR}' class='avatar'/><div><div class='bubble bot'><span class='dots'><span class='dot'></span><span class='dot'></span><span class='dot'></span></span></div><div class='meta'>🤖 {traducir_ui('Trevly — escribiendo...')}</div></div></div>",
            unsafe_allow_html=True
        )

        with st.spinner(traducir_ui("Procesando tu mensaje... 💭")):
            bot_resp, resultado = procesar_mensaje_usuario(user_input)
            bot_resp = traducir_ui(bot_resp)
            st.session_state.chat_sessions[st.session_state.active_chat]["resultado"] = resultado
            st.session_state.resultado = resultado

        placeholder.empty()
        history.append({
            "role":"bot",
            "content":bot_resp,
            "ts":datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Mostrar mapa y recomendaciones
        res = resultado
        try:
            m = folium.Map(location=res["origen"], zoom_start=12)
            folium.Marker(location=res["origen"], tooltip=traducir_ui("Tu ubicación"), icon=folium.Icon(color="blue")).add_to(m)
            for s in res["sitios_ranked"]:
                coord = [s.get("latitud", 3.437), s.get("longitud", -76.529)]
                folium.Marker(location=coord, tooltip=s.get("nombre_google", s.get("nombre"))).add_to(m)
            st_folium(m, width=700, height=400)
        except Exception as e:
            st.error(f"{traducir_ui('Error al renderizar mapa')}: {e}")

        st.markdown(traducir_ui("### 🌟 Recomendaciones finales:"))
        for i, s in enumerate(res["sitios_ranked"], start=1):
            st.markdown(f"**{i}. {s.get('nombre_google', s.get('nombre', traducir_ui('Sitio')))}** — {s.get('categoria','')}")

        st.markdown(traducir_ui("### 🤖 Recomendación personalizada:"))
        st.markdown(traducir_ui(res["recomendacion_rag"]))


# -------------------- Mostrar resultado guardado si existe --------------------
if st.session_state.chat_sessions[st.session_state.active_chat].get("resultado") and st.session_state.editing is None:
    st.markdown("---")
    st.markdown(traducir_ui("### Mapa y recomendaciones guardadas para este chat"))
    res = st.session_state.chat_sessions[st.session_state.active_chat]["resultado"]

    if res.get("recomendacion_rag"):
        res["recomendacion_rag"] = traducir_ui(res["recomendacion_rag"])

    try:
        m2 = folium.Map(location=res["origen"], zoom_start=12)
        folium.Marker(location=res["origen"], tooltip=traducir_ui("Tu ubicación"), icon=folium.Icon(color="blue")).add_to(m2)
        for s in res["sitios_ranked"]:
            coord = [s.get("latitud", 3.437), s.get("longitud", -76.529)]
            folium.Marker(location=coord, tooltip=s.get("nombre_google", s.get("nombre"))).add_to(m2)
        st_folium(m2, width=700, height=400)
    except Exception as e:
        st.error(f"{traducir_ui('Error al renderizar mapa guardado')}: {e}")

    st.markdown(traducir_ui("### 🌟 Recomendaciones guardadas:"))
    for i, s in enumerate(res["sitios_ranked"], start=1):
        st.markdown(f"**{i}. {s.get('nombre_google', s.get('nombre', traducir_ui('Sitio')))}** — {s.get('categoria','')}")
    st.markdown(traducir_ui("### 🤖 Recomendación personalizada guardada:"))
    st.markdown(res["recomendacion_rag"])
