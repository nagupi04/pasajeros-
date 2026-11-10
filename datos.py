import pandas as pd
import os

# Dirección del archivo en internet
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

# Dirección del archivo local (copia de respaldo para cuando no hay internet)
ARCHIVO_LOCAL = "datos/airline-passengers.csv"


def cargar_datos():
    """
    Intenta descargar los datos de pasajeros desde internet.
    Si no hay conexión, los lee desde la carpeta 'datos/'.
    Devuelve un DataFrame con columnas: Month, Passengers.
    """
    try:
        # Intento 1: bajar el archivo desde internet
        df = pd.read_csv(URL)
        print("Datos cargados desde internet.")

        # Intentar guardar copia local (solo funciona en entornos con permisos de escritura)
        try:
            os.makedirs("datos", exist_ok=True)
            df.to_csv(ARCHIVO_LOCAL, index=False)
        except Exception:
            pass  # En la nube el sistema de archivos puede ser de solo lectura; no es un error

    except Exception:
        # Intento 2: leer desde la copia local (disponible si el repo la incluye)
        print("Sin internet. Cargando datos desde archivo local.")
        df = pd.read_csv(ARCHIVO_LOCAL)

    # Convertir la columna Month a tipo fecha (ej: "1949-01" -> fecha real)
    df["Month"] = pd.to_datetime(df["Month"])

    # Asegurarnos de que Passengers sea número entero
    df["Passengers"] = df["Passengers"].astype(int)

    return df
