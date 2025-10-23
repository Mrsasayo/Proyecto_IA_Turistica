# src/mapa_rutas.py
import folium
from db import geodatos_col

def generar_mapa(origen, sitios):
    """
    Genera un mapa con marcadores y ruta aproximada.
    origen: [lat, lon]
    sitios: lista de diccionarios con '_id' y 'nombre'
    """
    mapa = folium.Map(location=origen, zoom_start=13)
    
    # Marcador de origen
    folium.Marker(location=origen, tooltip="Tu ubicación", icon=folium.Icon(color="blue")).add_to(mapa)
    
    for s in sitios:
        geo = geodatos_col.find_one({"sitio_id": s['_id']})
        if geo:
            folium.Marker([geo['lat'], geo['lon']], tooltip=f"{s['nombre']}\nTiempo aproximado: {geo.get('tiempo_estimado', 'N/A')} mins").add_to(mapa)
    
    return mapa
