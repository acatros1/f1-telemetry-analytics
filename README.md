# Analizador de Ritmo y Predicción en F1 🏎️

Este proyecto es una herramienta de análisis y visualización de datos de Fórmula 1 que procesa telemetría oficial de las sesiones para evaluar el rendimiento de los equipos y pilotos en tandas largas, estimar la degradación de los neumáticos y proyectar el balance del fin de semana.

La aplicación cuenta con una interfaz gráfica interactiva que facilita la exploración de los datos sin necesidad de interactuar directamente con la línea de comandos.

## Características principales

*   **Procesamiento de Telemetría:** Carga de datos de tiempos por vuelta a través de la API de FastF1 con sistema de caché local para optimizar el consumo de red.
*   **Filtrado de Outliers:** Remoción automatizada de vueltas no representativas (vueltas de entrada/salida de boxes y anomalías estadísticas en carrera) para obtener promedios limpios.
*   **Visualización Interactiva:** Tableros dinámicos desarrollados con Plotly para comparar ritmos de carrera ordenando por mediana, percentiles o tamaño de caja (IQR) para medir la consistencia del piloto.
*   **Cálculo de Degradación:** Estimación lineal de la pérdida de rendimiento por vuelta utilizando regresión lineal sobre la vida del neumático.
*   **Análisis Multisesión:** Comparativa de evolución de tiempos desde los entrenamientos libres (FP1, FP2, FP3) hasta la sesión de Clasificación.

## Tecnologías utilizadas

*   **Lenguaje de programación:** Python 3.x
*   **Análisis y procesamiento de datos:** Pandas, NumPy, Scikit-learn (LOWESS smoothing)
*   **Visualización gráfica:** Plotly Express / Graph Objects, Matplotlib
*   **Interfaz de usuario (Dashboard):** Streamlit
*   **Acceso a datos deportivos:** FastF1 API

## Instalación y Configuración

Para ejecutar este proyecto en tu entorno local, sigue estos pasos:

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/f1-telemetry-analytics.git
    cd f1-telemetry-analytics
    ```

2.  **Crear y activar un entorno virtual (opcional pero recomendado):**
    ```bash
    python3 -m venv f1_env
    source f1_env/bin/activate  # En Linux/macOS
    # f1_env\Scripts\activate  # En Windows
    ```

3.  **Instalar las dependencias necesarias:**
    ```bash
    pip install -r requirements.txt
    ```

## Instrucciones de uso

El proyecto cuenta con dos scripts principales que puedes ejecutar según tus necesidades:

### 1. Extractor de consola (CLI)
Para realizar análisis rápidos en terminal y generar tablas formateadas en Markdown:
```bash
python extractor.py

2. Dashboard interactivo (Web)

Para iniciar la interfaz interactiva con gráficos y tableros dinámicos en tu navegador web:
code Bash

streamlit run f1-pre.py

Estructura del proyecto

    f1-pre.py: Código principal del tablero de control interactivo desarrollado en Streamlit.

    extractor.py: Script de consola diseñado para extraer datos de ritmos corregidos por combustible.

    requirements.txt: Archivo con la lista de librerías necesarias para el funcionamiento del proyecto.

    .gitignore: Archivo de configuración para evitar la subida de entornos virtuales y archivos de caché de datos masivos.

