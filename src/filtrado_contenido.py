# src/filtrado_contenido.py
from sentence_transformers import SentenceTransformer, util
import torch
from db import sitios_col  # la colección de sitios

def calcular_score_contenido(usuario, model):
    """
    Calcula similitud de embeddings entre preferencias del usuario y los sitios almacenados en MongoDB.
    
    Args:
        usuario (dict): Diccionario con datos del usuario, incluyendo 'intereses'.
        model (SentenceTransformer): Modelo para generar embeddings.
    
    Returns:
        dict: Diccionario {_id_sitio: score_normalizado}.
    """
    if not usuario or not usuario.get('intereses'):
        return {}

    # Concatenamos intereses del usuario como texto
    preferencias_texto = " ".join(usuario['intereses'])
    query_embedding = model.encode(preferencias_texto, convert_to_tensor=True)

    # Traer sitios con embedding
    sitios = list(sitios_col.find({"embedding": {"$exists": True}}))
    if not sitios:
        return {}

    # Crear tensor de embeddings de sitios
    sitio_embeddings = torch.tensor([s['embedding'] for s in sitios])

    # Calcular similitud coseno
    cos_scores = util.cos_sim(query_embedding, sitio_embeddings)[0]

    # Crear diccionario {_id_sitio: score}
    score_contenido = {sitios[i]['_identificación']: float(cos_scores[i]) for i in range(len(sitios))}

    # Normalizar
    max_score = max(score_contenido.values()) if score_contenido else 1
    score_contenido = {k: v / max_score for k, v in score_contenido.items()}

    return score_contenido
