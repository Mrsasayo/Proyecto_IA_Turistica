# src/filtrado_colaborativo.py
from db import usuarios_col, estadisticas_col
import pandas as pd
import numpy as np

def calcular_score_colaborativo(usuario, sitios, cluster, n_top=10):
    """
    Calcula score colaborativo usando historial de usuarios del mismo cluster.
    Ajustado a MongoDB con '_identificación' en lugar de 'sitio_id'.
    """
    # Filtrar usuarios del mismo cluster
    usuarios_cluster = list(usuarios_col.find({"cluster": cluster}))

    # Crear un DataFrame de historial
    historial = list(estadisticas_col.find())
    df_hist = pd.DataFrame(historial)

    # --- Si no existe la columna, crearla vacía para evitar errores ---
    if '_identificación' not in df_hist.columns:
        df_hist['_identificación'] = np.nan
    if 'usuario_id' not in df_hist.columns:
        df_hist['usuario_id'] = np.nan
    if 'rating' not in df_hist.columns:
        df_hist['rating'] = 0

    # Filtrar solo sitios candidatos
    ids_sitios = [s['_identificación'] for s in sitios if '_identificación' in s]
    df_hist = df_hist[df_hist['_identificación'].isin(ids_sitios)]

    # Filtrar historial solo de usuarios del cluster
    ids_usuarios = [u['_id'] for u in usuarios_cluster if '_id' in u]
    df_cluster = df_hist[df_hist['usuario_id'].isin(ids_usuarios)]

    # Calcular promedio de rating por sitio
    score_colab = df_cluster.groupby('_identificación')['rating'].mean().to_dict()

    # Normalizar scores
    max_score = max(score_colab.values()) if score_colab else 1
    score_colab = {k: v/max_score for k, v in score_colab.items()}

    return score_colab
