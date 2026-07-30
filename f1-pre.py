import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================
# CONFIGURACIÓN Y MAPAS ESTÁTICOS F1 2025
# ============================================
OFFICIAL_TEAM_COLORS_2025 = {
    'Red Bull Racing': '#3671C6',
    'Red Bull': '#3671C6',
    'RB': '#6692FF',
    'Visa Cash App RB': '#6692FF',
    'Ferrari': '#F91536',
    'Mercedes': '#27F4D2',
    'McLaren': '#FF8000',
    'Aston Martin': '#229971',
    'Alpine': '#FF87BC',
    'Williams': '#64C4FF',
    'Haas F1 Team': '#B6BABD',
    'Haas': '#B6BABD',
    'Kick Sauber': '#52E252',
    'Sauber': '#52E252'
}

DRIVER_TEAM_2025 = {
    'VER': 'Red Bull Racing',   'PER': 'Red Bull Racing',
    'LEC': 'Ferrari',           'HAM': 'Ferrari',
    'RUS': 'Mercedes',          'ANT': 'Mercedes',
    'NOR': 'McLaren',           'PIA': 'McLaren',
    'ALO': 'Aston Martin',      'STR': 'Aston Martin',
    'GAS': 'Alpine',            'DOO': 'Alpine',
    'ALB': 'Williams',          'SAI': 'Williams',
    'TSU': 'RB',                'HAD': 'RB',
    'OCO': 'Haas',              'BEA': 'Haas',
    'HUL': 'Kick Sauber',       'BOR': 'Kick Sauber'
}

# Silenciar logs internos de FastF1 para evitar ruido en la app
fastf1.logger.set_log_level('ERROR')

# ============================================
# FUNCIONES DE PROCESAMIENTO Y UTILERÍA
# ============================================
def setup_cache():
    cache_dir = 'f1_cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    try:
        fastf1.Cache.enable_cache(cache_dir)
    except AttributeError:
        try:
            fastf1.enable_cache(cache_dir)
        except Exception:
            pass

def darken_color(hex_color, factor=0.75):
    hex_color = hex_color.replace('#', '').strip()
    if len(hex_color) != 6:
        return '#808080'
    try:
        r = max(0, int(int(hex_color[0:2], 16) * factor))
        g = max(0, int(int(hex_color[2:4], 16) * factor))
        b = max(0, int(int(hex_color[4:6], 16) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return '#808080'

def get_team_color(team_name, driver_abbr=None, session=None):
    team_lower = team_name.lower().strip() if team_name else ''
    for key, color in OFFICIAL_TEAM_COLORS_2025.items():
        if key.lower() in team_lower or team_lower in key.lower():
            return color
    if session:
        try:
            import fastf1.plotting
            color = fastf1.plotting.get_team_color(team_name, session=session)
            if color:
                return color
        except Exception:
            pass
    if driver_abbr:
        driver_team = DRIVER_TEAM_2025.get(driver_abbr.upper())
        if driver_team:
            for key, color in OFFICIAL_TEAM_COLORS_2025.items():
                if key.lower() in driver_team.lower():
                    return color
    return '#808080'

def load_session_safe(year, gp, session_name):
    try:
        session = fastf1.get_session(year, gp, session_name)
        session.load(telemetry=False, weather=False, messages=False)
        return session
    except Exception:
        return None

# ============================================
# LOGICA DE EXTRACCIÓN DE DATOS (CON OPTIMIZACIÓN DE CACHÉ DE STREAMLIT)
# ============================================
@st.cache_data(show_spinner=False)
def get_all_race_data(year, gp):
    setup_cache()

    # 1. PRÁCTICAS
    prac_laps_list = []
    sessions_to_load = ['FP1', 'FP2', 'FP3']
    for fp in sessions_to_load:
        session = load_session_safe(year, gp, fp)
        if session:
            try:
                laps_all = session.laps
                laps_clean = laps_all[laps_all['PitOutTime'].isna() & laps_all['PitInTime'].isna() & laps_all['LapTime'].notna()].copy()
                filtered_laps = []
                for driver in laps_clean['Driver'].unique():
                    driver_laps = laps_clean[laps_clean['Driver'] == driver].copy()
                    if len(driver_laps) >= 3:
                        best_lap = driver_laps['LapTime'].dt.total_seconds().min()
                        driver_laps = driver_laps[driver_laps['LapTime'].dt.total_seconds() < (best_lap * 1.08)]
                        filtered_laps.append(driver_laps)
                if filtered_laps:
                    laps = pd.concat(filtered_laps)
                    laps['Session_Type'] = fp
                    laps['Session_Order'] = sessions_to_load.index(fp)
                    prac_laps_list.append(laps)
            except Exception:
                pass

    # Alternativa Sprint si no hay FP
    if not prac_laps_list:
        session = load_session_safe(year, gp, 'S')
        if session:
            try:
                laps_all = session.laps
                laps_clean = laps_all[laps_all['PitOutTime'].isna() & laps_all['PitInTime'].isna() & laps_all['LapTime'].notna()].copy()
                filtered_laps = []
                for driver in laps_clean['Driver'].unique():
                    driver_laps = laps_clean[laps_clean['Driver'] == driver].copy()
                    if len(driver_laps) >= 3:
                        best_lap = driver_laps['LapTime'].dt.total_seconds().min()
                        driver_laps = driver_laps[driver_laps['LapTime'].dt.total_seconds() < (best_lap * 1.08)]
                        filtered_laps.append(driver_laps)
                if filtered_laps:
                    laps = pd.concat(filtered_laps)
                    laps['Session_Type'] = 'Sprint'
                    laps['Session_Order'] = 0
                    prac_laps_list.append(laps)
            except Exception:
                pass

    all_prac_laps = pd.concat(prac_laps_list) if prac_laps_list else pd.DataFrame()
    if not all_prac_laps.empty:
        all_prac_laps['LapTime_sec'] = all_prac_laps['LapTime'].dt.total_seconds()
        prac_stats = all_prac_laps.groupby('Driver').agg(
            ritmo_mediana=('LapTime_sec', 'median'),
            consistencia_std=('LapTime_sec', 'std')
        ).reset_index()
        prac_stats['ritmo_rank'] = prac_stats['ritmo_mediana'].rank()
        prac_stats['cons_rank'] = prac_stats['consistencia_std'].rank()
    else:
        prac_stats = pd.DataFrame()

    # 2. SPRINT QUALY
    sq_df = pd.DataFrame()
    sq_session = load_session_safe(year, gp, 'SQ')
    if sq_session:
        results = sq_session.results
        sq_df = results[['Abbreviation', 'Position']].rename(columns={'Abbreviation': 'Driver', 'Position': 'sq_rank'})
        try:
            sq_laps = sq_session.laps.pick_quicklaps().copy()
            sq_laps['LapTime_sec'] = sq_laps['LapTime'].dt.total_seconds()
            best_sq_times = sq_laps.groupby('Driver')['LapTime_sec'].min().reset_index()
            sq_df = pd.merge(sq_df, best_sq_times, on='Driver', how='left')
        except Exception:
            pass

    # 3. SPRINT RACE
    s_results = pd.DataFrame()
    s_laps = pd.DataFrame()
    s_session = load_session_safe(year, gp, 'S')
    if s_session:
        results = s_session.results
        s_results = results[['Abbreviation', 'Position', 'Points']].rename(columns={'Abbreviation': 'Driver', 'Position': 's_result_pos', 'Points': 's_points'})
        try:
            s_laps = s_session.laps.pick_quicklaps().copy()
            s_laps['LapTime_sec'] = s_laps['LapTime'].dt.total_seconds()
            best_s_times = s_laps.groupby('Driver')['LapTime_sec'].min().reset_index()
            s_results = pd.merge(s_results, best_s_times, on='Driver', how='left')
        except Exception:
            pass

    # 4. QUALIFYING Y COLORES
    qualy_df = pd.DataFrame()
    driver_colors = {}
    q_session = load_session_safe(year, gp, 'Q')

    active_session = None
    for session_name in ['Q', 'FP1', 'FP2', 'FP3', 'SQ']:
        session = load_session_safe(year, gp, session_name)
        if session and hasattr(session, 'results') and session.results is not None and not session.results.empty:
            active_session = session
            break

    if active_session is not None:
        team_counts = {}
        pilotos_encontrados = set()
        for _, row in active_session.results.iterrows():
            driver = row['Abbreviation']
            team = row['TeamName']
            if driver not in pilotos_encontrados:
                pilotos_encontrados.add(driver)
                color = get_team_color(team, driver_abbr=driver, session=active_session)
                if team not in team_counts:
                    team_counts[team] = 0
                    driver_colors[driver] = color
                else:
                    team_counts[team] += 1
                    driver_colors[driver] = darken_color(color, 0.70)

    for driver, team in DRIVER_TEAM_2025.items():
        if driver not in driver_colors:
            color = '#808080'
            for key, col in OFFICIAL_TEAM_COLORS_2025.items():
                if key.lower() in team.lower():
                    color = col
                    break
            driver_colors[driver] = color

    if q_session:
        qualy_df = q_session.results[['Abbreviation', 'Position']].rename(columns={'Abbreviation': 'Driver', 'Position': 'q_rank'})
        try:
            q_laps = q_session.laps.pick_quicklaps().copy()
            q_laps['LapTime_sec'] = q_laps['LapTime'].dt.total_seconds()
            best_q_times = q_laps.groupby('Driver')['LapTime_sec'].min().reset_index()
            qualy_df = pd.merge(qualy_df, best_q_times, on='Driver', how='left')
        except Exception:
            pass

    # 5. REAL RACE RESULTS
    actual_top_10 = []
    r_session = load_session_safe(year, gp, 'R')
    if r_session and r_session.results is not None and not r_session.results.empty:
        actual_results = r_session.results.dropna(subset=['Position']).sort_values('Position')
        actual_top_10 = actual_results.head(10)['Abbreviation'].tolist()

    return prac_stats, all_prac_laps, sq_df, s_results, s_laps, qualy_df, driver_colors, actual_top_10

# ============================================
# ALGORITMOS DE ORDENACIÓN Y MATEMÁTICAS PLOTLY
# ============================================
def predict_race_order(qualy_df, prac_stats):
    if prac_stats.empty:
        return pd.DataFrame()
    if qualy_df.empty:
        df = prac_stats.copy()
        df['pred_score'] = (0.75 * df['ritmo_rank']) + (0.25 * df['cons_rank'])
        return df.sort_values('pred_score').reset_index(drop=True)
    df = pd.merge(qualy_df, prac_stats, on='Driver', how='left')
    max_rank = df['q_rank'].max()
    if pd.isna(max_rank):
        max_rank = 20
    df['ritmo_rank'] = df['ritmo_rank'].fillna(max_rank)
    df['cons_rank'] = df['cons_rank'].fillna(max_rank)
    df['pred_score'] = (0.6 * df['q_rank']) + (0.3 * df['ritmo_rank']) + (0.1 * df['cons_rank'])
    return df.sort_values('pred_score').reset_index(drop=True)

def compute_degradation(prac_laps):
    if prac_laps.empty:
        return pd.DataFrame()
    degradation_list = []
    for driver in prac_laps['Driver'].unique():
        driver_laps = prac_laps[prac_laps['Driver'] == driver].dropna(subset=['LapTime_sec', 'LapNumber'])
        if len(driver_laps) < 4:
            continue
        q1 = driver_laps['LapTime_sec'].quantile(0.25)
        q3 = driver_laps['LapTime_sec'].quantile(0.75)
        iqr = q3 - q1
        driver_laps_filtered = driver_laps[driver_laps['LapTime_sec'] <= (q3 + 1.2 * iqr)]
        if len(driver_laps_filtered) < 3:
            continue
        x = driver_laps_filtered['LapNumber'].values.astype(float)
        y = driver_laps_filtered['LapTime_sec'].values.astype(float)
        try:
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            degradation_list.append({'Driver': driver, 'deg_slope': slope})
        except Exception:
            pass
    return pd.DataFrame(degradation_list).sort_values('deg_slope') if degradation_list else pd.DataFrame()

def plotly_quantile(y, quantile):
    y_sorted = sorted([val for val in y if not np.isnan(val)])
    n = len(y_sorted)
    if n == 0: return np.nan
    if n == 1: return y_sorted[0]
    x = n * quantile + 0.5
    x1 = max(1, min(n, int(np.floor(x))))
    x2 = max(1, min(n, int(np.ceil(x))))
    y1 = y_sorted[x1 - 1]
    y2 = y_sorted[x2 - 1]
    return y1 if x1 == x2 else y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)

def get_plotly_quartile_order(df, col_x, col_y, quantile):
    order_dict = {}
    for group_name, group_df in df.groupby(col_x):
        values = group_df[col_y].dropna().values
        order_dict[group_name] = plotly_quantile(values, quantile)
    return sorted(order_dict.keys(), key=lambda k: (np.isnan(order_dict[k]), order_dict[k]))

def get_plotly_fence_order(df, col_x, col_y, type_fence='lower'):
    order_dict = {}
    for group_name, group_df in df.groupby(col_x):
        values = group_df[col_y].dropna().values
        if len(values) == 0:
            order_dict[group_name] = np.nan
            continue
        q1 = plotly_quantile(values, 0.25)
        q3 = plotly_quantile(values, 0.75)
        iqr = q3 - q1
        if type_fence == 'lower':
            theoretical = q1 - 1.5 * iqr
            actual_points = values[values >= theoretical]
            order_dict[group_name] = np.min(actual_points) if len(actual_points) > 0 else np.min(values)
        else:
            theoretical = q3 + 1.5 * iqr
            actual_points = values[values <= theoretical]
            order_dict[group_name] = np.max(actual_points) if len(actual_points) > 0 else np.max(values)
    return sorted(order_dict.keys(), key=lambda k: (np.isnan(order_dict[k]), order_dict[k]))

def get_plotly_iqr_order(df, col_x, col_y):
    order_dict = {}
    for group_name, group_df in df.groupby(col_x):
        values = group_df[col_y].dropna().values
        if len(values) == 0:
            order_dict[group_name] = np.nan
            continue
        q1 = plotly_quantile(values, 0.25)
        q3 = plotly_quantile(values, 0.75)
        # El tamaño de la caja es la diferencia entre Q3 y Q1
        order_dict[group_name] = q3 - q1
    # Ordena de menor a mayor tamaño (los valores NaN se envían al final)
    return sorted(order_dict.keys(), key=lambda k: (np.isnan(order_dict[k]), order_dict[k]))

def generate_practice_boxplot(df, title, driver_colors):
    med_ord = get_plotly_quartile_order(df, 'Driver', 'LapTime_sec', 0.50)
    q1_ord = get_plotly_quartile_order(df, 'Driver', 'LapTime_sec', 0.25)
    q3_ord = get_plotly_quartile_order(df, 'Driver', 'LapTime_sec', 0.75)
    lf_ord = get_plotly_fence_order(df, 'Driver', 'LapTime_sec', 'lower')
    uf_ord = get_plotly_fence_order(df, 'Driver', 'LapTime_sec', 'upper')
    iqr_ord = get_plotly_iqr_order(df, 'Driver', 'LapTime_sec') # <--- Nueva ordenación por tamaño de caja

    fig = px.box(df, x='Driver', y='LapTime_sec', color='Driver',
                 title=title,
                 labels={'LapTime_sec': 'Tiempo por vuelta (segundos)'},
                 category_orders={"Driver": med_ord},
                 color_discrete_map=driver_colors)

    fig.update_layout(
        xaxis_categoryorder="array",
        xaxis_categoryarray=med_ord,
        updatemenus=[
            dict(
                active=0,
                type="dropdown",
                buttons=[
                    dict(label="Mediana", method="relayout", args=[{"xaxis.categoryarray": med_ord, "xaxis.categoryorder": "array", "title.text": f"{title} (ordenado por Mediana)"}]),
                    dict(label="Q1", method="relayout", args=[{"xaxis.categoryarray": q1_ord, "xaxis.categoryorder": "array", "title.text": f"{title} (ordenado por Q1)"}]),
                    dict(label="Q3", method="relayout", args=[{"xaxis.categoryarray": q3_ord, "xaxis.categoryorder": "array", "title.text": f"{title} (ordenado por Q3)"}]),
                    dict(label="Tamaño de Caja (Consistencia)", method="relayout", args=[{"xaxis.categoryarray": iqr_ord, "xaxis.categoryorder": "array", "title.text": f"{title} (ordenado por Tamaño de Caja - IQR)"}]), # <--- Nueva opción en el dropdown
                    dict(label="Lower Fence", method="relayout", args=[{"xaxis.categoryarray": lf_ord, "xaxis.categoryorder": "array", "title.text": f"{title} (ordenado por Lower Fence)"}]),
                    dict(label="Upper Fence", method="relayout", args=[{"xaxis.categoryarray": uf_ord, "xaxis.categoryorder": "array", "title.text": f"{title} (ordenado por Upper Fence)"}]),
                ],
                direction="down", showactive=True, x=0.01, xanchor="left", y=1.15, yanchor="top"
            )
        ]
    )
    return fig

# ============================================
# INTERFAZ DE USUARIO EN STREAMLIT
# ============================================
def main():
    st.set_page_config(page_title="F1 Analytics & Prediction Motor", layout="wide", page_icon="🏁")

    st.title("🏁 MOTOR DE PREDICCIÓN DE FÓRMULA 1 🏁")
    st.markdown("Análisis avanzado de ritmos en tandas largas, degradación y balance del fin de semana usando telemetría e históricos de **FastF1**.")

    # Control Sidebar
    st.sidebar.header("Configuración del Evento")
    year = st.sidebar.number_input("Temporada (Año)", min_value=2020, max_value=2026, value=2025, step=1)
    gp = st.sidebar.text_input("Gran Premio (ej. Monza, Bahrain, Monaco)", value="Monza")

    execute = st.sidebar.button("Iniciar Análisis Completo", use_container_width=True)

    if execute or 'f1_analyzed' in st.session_state:
        st.session_state['f1_analyzed'] = True

        with st.spinner("Cargando datos desde la API de FastF1... (La primera vez de cada circuito puede demorar un minuto)"):
            prac_stats, all_prac_laps, sq_df, s_results, s_laps, qualy_df, driver_colors, actual_top_10 = get_all_race_data(year, gp)

        if prac_stats.empty and qualy_df.empty:
            st.error("No se pudieron extraer datos válidos para este Gran Premio o año. Verifica los parámetros.")
            return

        # Advertencia de Qualy faltante si aplica
        if qualy_df.empty:
            st.warning("⚠️ No hay datos de Clasificación (Q) disponibles. La predicción se generará únicamente con sesiones de Práctica.")

        prediction_df = predict_race_order(qualy_df, prac_stats)

        # ----------------------------------------------------
        # DISTRIBUCIÓN DE CONTENIDO EN PESTAÑAS (TABS)
        # ----------------------------------------------------
        tab_pred, tab_ritmo, tab_qualy, tab_deg = st.tabs([
            "🔮 Predicción vs Realidad",
            "⏱️ Ritmo en Prácticas Libres",
            "🏎️ Clasificación y Sprint",
            "📈 Evolución y Degradación"
        ])

        # TAB 1: PREDICCIÓN
        with tab_pred:
            st.subheader("Predicción del Algoritmo del Fin de Semana")
            if not prediction_df.empty:
                predicted_top_10 = prediction_df['Driver'].head(10).tolist()

                col_list, col_plot = st.columns([1, 2])
                with col_list:
                    st.markdown("### 🏆 Top 10 Predicho")
                    for idx, driver in enumerate(predicted_top_10):
                        st.markdown(f"**{idx+1}.** `{driver}`")

                with col_plot:
                    if actual_top_10:
                        st.markdown("### 📊 Comparativa de Rendimiento Directo")
                        pred_top10_lim = prediction_df['Driver'].head(len(actual_top_10)).tolist()
                        actual_top_10_limited = actual_top_10[:len(pred_top10_lim)]
                        n_elements = len(pred_top10_lim)

                        df_compare = pd.DataFrame({
                            'Piloto': pred_top10_lim + actual_top_10_limited,
                            'Posición': list(range(1, n_elements + 1)) + list(range(1, n_elements + 1)),
                            'Estado': ['Predicción'] * n_elements + ['Realidad'] * n_elements
                        })
                        fig4 = px.line(df_compare, x='Estado', y='Posición', color='Piloto', markers=True,
                                       title='Evaluación del Algoritmo vs Top 10 Real de Carrera',
                                       color_discrete_map=driver_colors)
                        fig4.update_yaxes(autorange="reversed", tickvals=list(range(1, n_elements + 1)))
                        st.plotly_chart(fig4, use_container_width=True)

                        # Cálculo del puntaje exacto de acierto
                        puntos_f1 = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                        pts_obtenidos = sum(puntos_f1[i] for i in range(min(10, len(actual_top_10_limited))) if pred_top10_lim[i] == actual_top_10_limited[i])
                        st.metric(label="Puntos de Predicción Obtenidos (Escala F1 Exacta)", value=f"{pts_obtenidos} / 101 pts")
                    else:
                        st.info("La carrera real aún no se ha disputado o los resultados no están disponibles para calcular los aciertos.")
            else:
                st.info("No hay suficientes datos procesables para proyectar el clasificador.")

        # TAB 2: RITMOS
        with tab_ritmo:
            st.subheader("Análisis de Ritmo Sostenido (Limpieza de Outliers e In/Out laps)")
            if not all_prac_laps.empty:
                # Boxplot Combinado Global
                q1_global = all_prac_laps['LapTime_sec'].quantile(0.25)
                q3_global = all_prac_laps['LapTime_sec'].quantile(0.75)
                iqr_global = q3_global - q1_global
                laps_filtered_global = all_prac_laps[all_prac_laps['LapTime_sec'] <= (q3_global + 1.5 * iqr_global)]

                fig_g = generate_practice_boxplot(laps_filtered_global, "Ritmo Combinado del Fin de Semana (FP1 + FP2 + FP3)", driver_colors)
                st.plotly_chart(fig_g, use_container_width=True)

                # Desglose de Prácticas Individuales
                st.markdown("---")
                st.subheader("Distribución Detallada por Sesión")
                sorted_sessions = all_prac_laps.sort_values('Session_Order')['Session_Type'].unique() if 'Session_Order' in all_prac_laps.columns else sorted(all_prac_laps['Session_Type'].unique())

                for session_name in sorted_sessions:
                    session_laps = all_prac_laps[all_prac_laps['Session_Type'] == session_name]
                    if not session_laps.empty:
                        q1_s = session_laps['LapTime_sec'].quantile(0.25)
                        q3_s = session_laps['LapTime_sec'].quantile(0.75)
                        iqr_s = q3_s - q1_s
                        session_filtered = session_laps[session_laps['LapTime_sec'] <= (q3_s + 1.5 * iqr_s)]

                        fig_s = generate_practice_boxplot(session_filtered, f"Ritmo Analítico - {session_name}", driver_colors)
                        st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.info("No se dispone de telemetría de vueltas limpias en prácticas.")

        # TAB 3: CLASIFICACIÓN Y SPRINT
        with tab_qualy:
            col_q, col_sq = st.columns(2)

            with col_q:
                st.subheader("Resultados de Clasificación Principal (Q)")
                if not qualy_df.empty:
                    qualy_df_sorted = qualy_df.sort_values('LapTime_sec')
                    fig_q = px.bar(qualy_df_sorted, x='Driver', y='LapTime_sec', color='Driver',
                                   title='Mejor Vuelta Absoluta en Q',
                                   labels={'LapTime_sec': 'Segundos'},
                                   category_orders={"Driver": qualy_df_sorted['Driver'].tolist()},
                                   color_discrete_map=driver_colors)
                    fig_q.update_yaxes(range=[qualy_df['LapTime_sec'].min() * 0.98, qualy_df['LapTime_sec'].max() * 1.02])
                    st.plotly_chart(fig_q, use_container_width=True)
                else:
                    st.info("No se registran datos en Clasificación principal.")

            with col_sq:
                st.subheader("Clasificación de Sprint (SQ)")
                if not sq_df.empty:
                    sq_df_sorted = sq_df.sort_values('LapTime_sec')
                    fig_sq = px.bar(sq_df_sorted, x='Driver', y='LapTime_sec', color='Driver',
                                    title='Mejor Vuelta en Clasificación Corta (SQ)',
                                    labels={'LapTime_sec': 'Segundos'},
                                    category_orders={"Driver": sq_df_sorted['Driver'].tolist()},
                                    color_discrete_map=driver_colors)
                    fig_sq.update_yaxes(range=[sq_df['LapTime_sec'].min() * 0.98, sq_df['LapTime_sec'].max() * 1.02])
                    st.plotly_chart(fig_sq, use_container_width=True)
                else:
                    st.info("Este fin de semana no contó con formato Sprint o los datos de Clasificación Sprint no están disponibles.")

            if not s_results.empty:
                st.markdown("---")
                st.subheader("Resultados e Impacto de Puntos de la Carrera Sprint")
                fig_sr = px.bar(s_results.head(10), x='Driver', y='s_points', color='Driver',
                               title='Puntos Otorgados en el Sprint (Top 10)',
                               labels={'s_points': 'Puntos'},
                               color_discrete_map=driver_colors)
                st.plotly_chart(fig_sr, use_container_width=True)

        # TAB 4: EVOLUCIÓN Y DEGRADACIÓN
        with tab_deg:
            st.subheader("Degradación Estimada y Variación de Tiempos")
            degradation_df = compute_degradation(all_prac_laps)

            if not degradation_df.empty:
                fig_deg = px.bar(degradation_df, x='Driver', y='deg_slope', color='Driver',
                                 title='Ranking de Degradación de Neumáticos (Pendiente Lineal del Ritmo)',
                                 labels={'deg_slope': 'Pérdida de Tiempo por Vuelta (segundos)', 'Driver': 'Piloto'},
                                 category_orders={"Driver": degradation_df['Driver'].tolist()},
                                 color_discrete_map=driver_colors)
                fig_deg.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
                st.plotly_chart(fig_deg, use_container_width=True)
            else:
                st.info("Muestra de vueltas continuas insuficiente para calcular tendencias de degradación lineal.")

            st.markdown("---")
            st.subheader("Evolución Cronológica del Fin de Semana")
            evolution_data = []
            if not all_prac_laps.empty:
                p_data = all_prac_laps.groupby(['Driver', 'Session_Type'])['LapTime_sec'].min().reset_index()
                p_data['Time_Order'] = p_data['Session_Type'].map({'FP1': 1, 'FP2': 2, 'FP3': 3, 'Sprint': 6})
                evolution_data.append(p_data)
            if not qualy_df.empty:
                q_data = qualy_df[['Driver', 'LapTime_sec']].copy()
                q_data['Session_Type'] = 'Clasificación'
                q_data['Time_Order'] = 4
                evolution_data.append(q_data)
            if not sq_df.empty:
                sq_data = sq_df[['Driver', 'LapTime_sec']].copy()
                sq_data['Session_Type'] = 'Clasificación Sprint'
                sq_data['Time_Order'] = 5
                evolution_data.append(sq_data)
            if not s_laps.empty:
                s_data = s_laps.groupby('Driver')['LapTime_sec'].min().reset_index()
                s_data['Session_Type'] = 'Sprint'
                s_data['Time_Order'] = 6
                evolution_data.append(s_data)

            if evolution_data:
                all_evolution = pd.concat(evolution_data, ignore_index=True).sort_values(by=['Driver', 'Time_Order'])
                fig_evo = px.line(all_evolution, x='Session_Type', y='LapTime_sec', color='Driver', markers=True,
                                  title='Evolución del Mejor Tiempo por Sesión',
                                  labels={'LapTime_sec': 'Segundos', 'Session_Type': 'Sesión'},
                                  color_discrete_map=driver_colors)
                fig_evo.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': ['FP1', 'FP2', 'FP3', 'Clasificación', 'Clasificación Sprint', 'Sprint']})
                st.plotly_chart(fig_evo, use_container_width=True)

                # Scatterplot Detallado con Tendencias LOWESS
                st.markdown("---")
                st.subheader("Comportamiento del Ritmo por Vuelta Sostenida")
                fig_scat = px.scatter(all_prac_laps, x='LapNumber', y='LapTime_sec', color='Driver',
                                      facet_col='Session_Type', facet_col_wrap=3, trendline="lowess",
                                      title='Curva Estructural de Rendimiento (LOWESS Smoother)',
                                      labels={'LapTime_sec': 'Tiempo de Vuelta (s)', 'LapNumber': 'Número de Vuelta'},
                                      color_discrete_map=driver_colors)
                fig_scat.update_layout(height=550)
                st.plotly_chart(fig_scat, use_container_width=True)

if __name__ == "__main__":
    main()
