import os
import warnings
import fastf1 as ff1
import pandas as pd
import numpy as np

# Silenciar advertencias de FastF1 y Pandas en consola
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 1. Configurar caché local
os.makedirs('fastf1_cache', exist_ok=True)
ff1.Cache.enable_cache('fastf1_cache')

print("="*50)
print("     EXTRACTOR INTERACTIVO DE RITMO DE CARRERA     ")
print("="*50)

# 2. Entrada de datos interactiva del usuario
try:
    year = int(input("1. Introduce el año de la temporada (ej. 2024): "))

    print("\nObteniendo el calendario de la temporada...")
    schedule = ff1.get_event_schedule(year, include_testing=False)

    # Filtrar eventos que no corresponden a Grandes Premios de la temporada (como entrenamientos de pretemporada)
    schedule = schedule[schedule['EventFormat'] != 'testing']

    if schedule.empty:
        raise ValueError(f"No se encontraron eventos para el año {year}.")

    print("\nGrandes Premios disponibles para esta temporada:")
    gps_list = []
    for idx, row in schedule.iterrows():
        gps_list.append(row)
        print(f" [{len(gps_list)}] {row['EventName']} ({row['Location']}, {row['Country']})")

    # Bucle de validación para la selección del circuito
    while True:
        try:
            gp_choice = int(input(f"\nSelecciona el número del Gran Premio (1-{len(gps_list)}): "))
            if 1 <= gp_choice <= len(gps_list):
                selected_event = gps_list[gp_choice - 1]
                gp_name = selected_event['EventName']
                break
            else:
                print(f"Por favor, selecciona un número válido entre 1 y {len(gps_list)}.")
        except ValueError:
            print("Por favor, introduce un número entero válido.")

    event = selected_event
    print(f"\n[OK] Evento seleccionado: {event['EventName']}")
    print(f"     Ubicación: {event['Location']}, {event['Country']}")
    print("-" * 50)

    # 3. Detectar y listar sesiones del fin de semana
    sessions_available = {}
    print("Sesiones programadas para este fin de semana:")
    for i in range(1, 6):
        s_name = event[f'Session{i}']
        if s_name:
            sessions_available[i] = s_name
            print(f" [{i}] {s_name}")

    # 4. Lógica de recomendación según el formato del GP
    recommendation = ""
    sprint_num = None
    fp2_num = None
    for num, name in sessions_available.items():
        if name == 'Sprint':
            sprint_num = num
        elif name in ['Practice 2', 'Second Practice', 'FP2']:
            fp2_num = num

    if sprint_num:
        recommendation = (
            f"\n💡 RECOMENDACIÓN DE ESTRATEGIA:\n"
            f" Este es un FIN DE SEMANA SPRINT. No hay simulación tradicional de FP2.\n"
            f" Te sugiero seleccionar la opción [{sprint_num}] (Sprint).\n"
            f" Al ser una carrera corta de 100km, es la simulación de ritmo real más pura posible."
        )
    elif fp2_num:
        recommendation = (
            f"\n💡 RECOMENDACIÓN DE ESTRATEGIA:\n"
            f" Este es un fin de semana TRADICIONAL.\n"
            f" Te sugiero seleccionar la opción [{fp2_num}] ({sessions_available[fp2_num]})\n"
            f" para analizar los long runs con alta carga de combustible."
        )
    else:
        recommendation = "\n💡 RECOMENDACIÓN: Selecciona la sesión de Practice o Race que prefieras."

    print(recommendation)
    print("-" * 50)

    # 5. Cargar sesión elegida
    choice = int(input("Selecciona el número de sesión para el análisis: "))
    selected_session_name = sessions_available[choice]

    print(f"\nCargando datos locales para {selected_session_name}...")
    session = ff1.get_session(year, gp_name, selected_session_name)
    session.load()

    # 6. FILTRADO INTELIGENTE DE VUELTAS (Sin pick_quicklaps)
    laps_all = session.laps
    laps_clean = laps_all[laps_all['PitOutTime'].isna() & laps_all['PitInTime'].isna() & laps_all['LapTime'].notna()]

    summary_data = []

    # Iterar de manera segura usando session.drivers
    for driver_num in session.drivers:
        driver_info = session.get_driver(driver_num)
        driver_abb = driver_info['Abbreviation']

        if not driver_abb:
            continue

        driver_laps = laps_clean.pick_drivers(driver_abb)

        if len(driver_laps) < 5:
            continue

        # FILTRO DINÁMICO DE OUTLIERS POR PILOTO:
        best_lap_sec = driver_laps['LapTime'].dt.total_seconds().min()
        driver_laps = driver_laps[driver_laps['LapTime'].dt.total_seconds() < (best_lap_sec * 1.08)]

        if len(driver_laps) < 5:
            continue

        # Aplicar la corrección de combustible
        adj_times = []
        for _, lap in driver_laps.iterrows():
            raw_time = lap['LapTime'].total_seconds()
            adj_time = raw_time - (0.03 * lap['TyreLife'])
            adj_times.append(adj_time)

        # Cálculos estadísticos finales
        median_raw = np.median([l.total_seconds() for l in driver_laps['LapTime']])
        median_adj = np.median(adj_times)

        q75_raw, q25_raw = np.percentile([l.total_seconds() for l in driver_laps['LapTime']], [75 ,25])
        iqr_raw = q75_raw - q25_raw

        q75_adj, q25_adj = np.percentile(adj_times, [75 ,25])
        iqr_adj = q75_adj - q25_adj

        max_life = driver_laps['TyreLife'].max()
        compounds = driver_laps['Compound'].unique().tolist()

        summary_data.append({
            'Piloto': driver_abb,
            'Vueltas': len(driver_laps),
            'Compuestos': "/".join(compounds),
            'Mediana_Raw': round(median_raw, 3),
            'Mediana_Corr': round(median_adj, 3),
            'IQR_Raw': round(iqr_raw, 3),
            'IQR_Corr': round(iqr_adj, 3),
            'Max_Vida_Goma': int(max_life)
        })

    # Crear DataFrame, ordenar de más rápido a más lento (Mediana Corregida)
    df_summary = pd.DataFrame(summary_data).sort_values(by='Mediana_Corr')

    # Imprimir en formato Markdown listo para copiar y pegar
    print("\n" + "="*50)
    print(f"ANÁLISIS DE RITMO DE CARRERA CORREGIDO: {event['EventName']} ({year})")
    print(f"SESIÓN ANALIZADA: {selected_session_name}")
    print("="*50)
    print("\n--- COPIA DESDE AQUÍ ABAJO Y PÉGALO EN EL CHAT ---")

    markdown_output = df_summary.to_markdown(index=False)
    print(markdown_output)

    # Guardar automáticamente en archivo de texto
    with open("resultado_analisis.txt", "w", encoding="utf-8") as f:
        f.write(f"GP: {event['EventName']} ({year}) | Sesión: {selected_session_name}\n\n")
        f.write(markdown_output)

    print(f"\n[OK] ¡Éxito! Se ha creado la tabla completa y se guardó en: 'resultado_analisis.txt'")
    print("Abre ese archivo con tu editor de texto favorito y cópiame el contenido.")

except Exception as e:
    print(f"\n[ERROR] Ocurrió un problema durante la extracción: {e}")
