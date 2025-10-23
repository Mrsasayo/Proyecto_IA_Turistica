# src/db.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# --- Cargar variables de entorno ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "turismo_cali")  # permite cambiar desde .env

# --- Conexión a la base de datos ---
def get_db():
    """Devuelve la conexión a la base de datos."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db

db = get_db()

# --- Función para obtener cualquier colección ---
def get_collection(name: str):
    """Devuelve la colección de la base de datos según su nombre."""
    return db[name]

# --- Conexiones a colecciones comunes ---
usuarios_col = get_collection("usuarios")
sitios_col = get_collection("sitios")
geodatos_col = get_collection("geodatos")
estadisticas_col = get_collection("estadisticas_turismo")

# --- Funciones de consulta ---
def obtener_usuarios():
    """Retorna todos los usuarios."""
    return list(usuarios_col.find())

def obtener_sitios():
    """Retorna todos los sitios."""
    return list(sitios_col.find())

def obtener_geodatos(sitio_id):
    """Retorna datos geográficos de un sitio específico."""
    return geodatos_col.find_one({"sitio_id": sitio_id})

def obtener_historial():
    """Retorna interacciones de usuarios con sitios (ratings, visitas)."""
    return list(estadisticas_col.find())
