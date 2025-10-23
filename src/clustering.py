# src/clustering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from db import usuarios_col

def cargar_datos_usuarios():
    usuarios = list(usuarios_col.find())
    df = pd.DataFrame(usuarios)
    return df


def entrenar_clustering(df, n_clusters=4):
    """Entrena un modelo KMeans para segmentar usuarios."""
    # Convertimos intereses en columnas one-hot
    df_intereses = df['intereses'].apply(lambda x: '|'.join(x) if isinstance(x, list) else "").str.get_dummies()
    
    # Seleccionamos features numéricas + intereses
    X = pd.concat([df[['edad','presupuesto']].fillna(0), df_intereses], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    
    # Map de clusters a perfiles
    perfil_map = {i: f"Perfil_{i}" for i in range(n_clusters)}
    return kmeans, scaler, perfil_map


def predecir_perfil(usuario, kmeans_model, scaler, perfil_map, columnas_intereses):
    """Predice el cluster y perfil del usuario."""
    # Manejar NaN y tipos
    edad = usuario.get('edad', np.nan)
    presupuesto = usuario.get('presupuesto', np.nan)
    if np.isnan(edad): edad = 0
    if np.isnan(presupuesto): presupuesto = 0

    # Procesar intereses
    intereses = usuario.get('intereses', [])
    intereses_str = '|'.join(intereses)

    # Corrección: usar Series para get_dummies()
    df_intereses = pd.Series([intereses_str]).str.get_dummies()

    # Asegurar columnas iguales a las del entrenamiento
    for col in columnas_intereses:
        if col not in df_intereses.columns:
            df_intereses[col] = 0
    df_intereses = df_intereses[columnas_intereses]

    # Concatenar features
    X = pd.concat([pd.DataFrame([[edad, presupuesto]], columns=['edad','presupuesto']), df_intereses], axis=1)
    X_scaled = scaler.transform(X)

    cluster = int(kmeans_model.predict(X_scaled)[0])  # ✅ Convertir a int nativo
    perfil = perfil_map[cluster]
    return cluster, perfil
