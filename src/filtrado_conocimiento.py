# src/filtrado_conocimiento.py
from db import sitios_col

def filtrar_por_usuario(usuario):
    """
    Filtra sitios según los intereses del usuario consultando MongoDB.
    """
    intereses = [i.lower() for i in usuario.get('intereses', []) if isinstance(i, str)]

    sitios = list(sitios_col.find())
    sitios_filtrados = []

    for s in sitios:
        nombre = str(s.get('nombre', '')).lower()
        categoria = str(s.get('categoria', '')).lower()
        descripcion = str(s.get('descripcion', '')).lower()

        match = any(
            interes in nombre or interes in categoria or interes in descripcion
            for interes in intereses
        )
        if match:
            sitios_filtrados.append(s)

    # Si no se encontró nada, devolver algunos sitios al azar
    if not sitios_filtrados and sitios:
        from random import sample
        sitios_filtrados = sample(sitios, min(10, len(sitios)))

    return sitios_filtrados
