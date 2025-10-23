from db import sitios_col

def ranking_final(sitios_ids, score_colab, score_contenido, w_colab=0.6, w_contenido=0.4):
    """
    Devuelve ranking final de sitios combinando scores colaborativo y contenido, 
    pero obteniendo la info directamente de MongoDB.
    """
    # Traer sitios desde MongoDB
    sitios = list(sitios_col.find({"_id": {"$in": sitios_ids}}))
    
    resultados = []
    for s in sitios:
        id_sitio = s['_id']
        final_score = w_colab*score_colab.get(id_sitio, 0) + w_contenido*score_contenido.get(id_sitio, 0)
        resultados.append({**s, "score_final": final_score})
    
    # Ordenar de mayor a menor
    resultados_ordenados = sorted(resultados, key=lambda x: x['score_final'], reverse=True)
    
    return resultados_ordenados
