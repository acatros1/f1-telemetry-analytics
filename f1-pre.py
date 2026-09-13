import os
import warnings

import fastf1
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)
fastf1.logger.set_log_level("ERROR")

# ============================================================
# CONFIGURACIÓN
# ============================================================
OFFICIAL_TEAM_COLORS_2025 = {
    "Red Bull Racing": "#3671C6", "Red Bull": "#3671C6",
    "RB": "#6692FF", "Visa Cash App RB": "#6692FF",
    "Ferrari": "#F91536", "Mercedes": "#27F4D2",
    "McLaren": "#FF8000", "Aston Martin": "#229971",
    "Alpine": "#0093CC", "Williams": "#64C4FF",
    "Haas F1 Team": "#B6BABD", "Haas": "#B6BABD",
    "Kick Sauber": "#52E252", "Sauber": "#52E252",
}

DRIVER_TEAM_2025 = {
    "VER": "Red Bull Racing", "TSU": "Red Bull Racing",
    "LEC": "Ferrari", "HAM": "Ferrari",
    "RUS": "Mercedes", "ANT": "Mercedes",
    "NOR": "McLaren", "PIA": "McLaren",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "GAS": "Alpine", "DOO": "Alpine",
    "ALB": "Williams", "SAI": "Williams",
    "HAD": "RB", "LAW": "RB",
    "OCO": "Haas", "BEA": "Haas",
    "HUL": "Kick Sauber", "BOR": "Kick Sauber",
}

DEFAULT_WEIGHTS = {
    "grid": 0.35,
    "qualifying": 0.25,
    "long_run": 0.25,
    "degradation": 0.10,
    "consistency": 0.05,
}

# ============================================================
# UTILIDADES
# ============================================================
def setup_cache():
    cache_dir = "f1_cache"
    os.makedirs(cache_dir, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(cache_dir)
    except AttributeError:
        try:
            fastf1.enable_cache(cache_dir)
        except Exception:
            pass


@st.cache_data(show_spinner=False)
def get_events_for_year(year):
    setup_cache()
    try:
        schedule = fastf1.get_event_schedule(year)
        events = schedule[schedule["RoundNumber"] > 0]
        if not events.empty:
            return events["EventName"].tolist()
    except Exception:
        pass

    return [
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
        "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
        "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Spanish Grand Prix",
        "Canadian Grand Prix", "Austrian Grand Prix", "British Grand Prix",
        "Belgian Grand Prix", "Hungarian Grand Prix", "Dutch Grand Prix",
        "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
        "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
        "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix",
    ]


def darken_color(hex_color, factor=0.70):
    value = str(hex_color).replace("#", "").strip()
    if len(value) != 6:
        return "#808080"
    try:
        r = int(int(value[0:2], 16) * factor)
        g = int(int(value[2:4], 16) * factor)
        b = int(int(value[4:6], 16) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "#808080"


def get_team_color(team_name, driver_abbr=None, session=None):
    team = str(team_name or "").lower().strip()
    for key, color in OFFICIAL_TEAM_COLORS_2025.items():
        key_lower = key.lower()
        if key_lower in team or team in key_lower:
            return color

    if session is not None:
        try:
            import fastf1.plotting
            color = fastf1.plotting.get_team_color(team_name, session=session)
            if color:
                return color
        except Exception:
            pass

    if driver_abbr:
        mapped_team = DRIVER_TEAM_2025.get(str(driver_abbr).upper())
        if mapped_team:
            for key, color in OFFICIAL_TEAM_COLORS_2025.items():
                if key.lower() in mapped_team.lower():
                    return color

    return "#808080"


def load_session_safe(year, gp, session_name):
    setup_cache()
    try:
        session = fastf1.get_session(year, gp, session_name)
        session.load(telemetry=False, weather=False, messages=False)
        return session
    except Exception:
        return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def rank_ascending(series):
    """1 = mejor para una métrica donde menor es mejor."""
    s = safe_numeric(series)
    return s.rank(method="average", ascending=True, na_option="bottom")


def normalize_rank(series):
    """Escala rank 0..1; 0 = mejor."""
    rank = rank_ascending(series)
    valid = rank.notna()
    n = valid.sum()
    if n <= 1:
        return pd.Series(0.0, index=series.index)
    return (rank - 1) / (n - 1)


def slope_and_residual_std(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4 or np.ptp(x) <= 0:
        return np.nan, np.nan
    try:
        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (slope * x + intercept)
        return float(slope), float(np.std(residuals, ddof=1))
    except Exception:
        return np.nan, np.nan

# ============================================================
# PREPARACIÓN DE VUELTAS Y STINTS
# ============================================================
def prepare_clean_laps(session, session_name):
    if session is None or not hasattr(session, "laps"):
        return pd.DataFrame()

    laps = session.laps.copy()
    if laps.empty:
        return pd.DataFrame()

    required = ["Driver", "LapTime"]
    if any(col not in laps.columns for col in required):
        return pd.DataFrame()

    laps["LapTime_sec"] = pd.to_timedelta(laps["LapTime"], errors="coerce").dt.total_seconds()
    laps = laps[laps["LapTime_sec"].notna()].copy()

    if "PitOutTime" in laps.columns:
        laps = laps[laps["PitOutTime"].isna()]
    if "PitInTime" in laps.columns:
        laps = laps[laps["PitInTime"].isna()]

    if "LapNumber" in laps.columns:
        laps["LapNumber"] = safe_numeric(laps["LapNumber"])
    if "Stint" in laps.columns:
        laps["Stint"] = safe_numeric(laps["Stint"])
    else:
        laps["Stint"] = np.nan
    if "TyreLife" in laps.columns:
        laps["TyreLife"] = safe_numeric(laps["TyreLife"])
    else:
        laps["TyreLife"] = np.nan
    if "Compound" in laps.columns:
        laps["Compound"] = laps["Compound"].fillna("Unknown").astype(str)
    else:
        laps["Compound"] = "Unknown"

    laps["Session_Type"] = session_name
    return laps


def extract_long_run_features(clean_laps, min_stint_laps=5):
    """
    Extrae ritmo de tandas largas por stint y compuesto.

    Para comparar pilotos sin mezclar compounds, se calcula un gap contra
    el mejor stint del mismo Session_Type + Compound. La degradación se
    estima dentro de cada stint sobre TyreLife cuando existe y presenta
    variación suficiente.
    """
    if clean_laps.empty:
        return pd.DataFrame(), pd.DataFrame()

    stint_rows = []
    group_cols = ["Driver", "Session_Type", "Stint", "Compound"]

    for keys, stint in clean_laps.groupby(group_cols, dropna=False):
        driver, session_type, stint_id, compound = keys
        stint = stint.sort_values("LapNumber").copy()

        if len(stint) < min_stint_laps:
            continue

        # Para evitar que una primera vuelta tras un pit stop contamine la
        # tendencia, la excluimos solo cuando hay suficientes vueltas.
        if len(stint) >= 7:
            analysis = stint.iloc[1:].copy()
        else:
            analysis = stint.copy()

        if len(analysis) < min_stint_laps:
            continue

        pace_median = float(analysis["LapTime_sec"].median())
        pace_q25 = float(analysis["LapTime_sec"].quantile(0.25))
        pace_q75 = float(analysis["LapTime_sec"].quantile(0.75))
        pace_iqr = pace_q75 - pace_q25

        tyre_x = analysis["TyreLife"]
        if tyre_x.notna().sum() >= 4 and tyre_x.nunique(dropna=True) >= 4:
            x = tyre_x.to_numpy(dtype=float)
            x_name = "TyreLife"
        else:
            x = analysis["LapNumber"].to_numpy(dtype=float)
            x_name = "LapNumber_fallback"

        slope, residual_std = slope_and_residual_std(
            x,
            analysis["LapTime_sec"].to_numpy(dtype=float),
        )

        stint_rows.append({
            "Driver": driver,
            "Session_Type": session_type,
            "Stint": stint_id,
            "Compound": compound,
            "Stint_Laps": len(analysis),
            "pace_median": pace_median,
            "pace_iqr": pace_iqr,
            "degradation_slope": slope,
            "consistency_residual_std": residual_std,
            "slope_axis": x_name,
        })

    stints = pd.DataFrame(stint_rows)
    if stints.empty:
        return stints, stints

    # Gap al mejor stint del mismo contexto.
    context = stints.groupby(["Session_Type", "Compound"], dropna=False)["pace_median"].transform("min")
    stints["pace_gap_sec"] = stints["pace_median"] - context

    # Peso proporcional al tamaño de la muestra, evitando que una tanda de 5
    # vueltas y una de 15 tengan la misma influencia.
    stints["weight"] = stints["Stint_Laps"].clip(lower=1)

    def weighted_median(values, weights):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        values, weights = values[mask], weights[mask]
        if len(values) == 0:
            return np.nan
        order = np.argsort(values)
        values, weights = values[order], weights[order]
        cutoff = weights.sum() / 2.0
        return float(values[np.searchsorted(np.cumsum(weights), cutoff)])

    driver_rows = []
    for driver, group in stints.groupby("Driver"):
        driver_rows.append({
            "Driver": driver,
            "long_run_pace_gap_sec": weighted_median(group["pace_gap_sec"], group["weight"]),
            "long_run_pace_raw_sec": weighted_median(group["pace_median"], group["weight"]),
            "degradation_slope": weighted_median(group["degradation_slope"], group["weight"]),
            "consistency_residual_std": weighted_median(group["consistency_residual_std"], group["weight"]),
            "long_run_stints": int(len(group)),
            "long_run_laps": int(group["Stint_Laps"].sum()),
        })

    features = pd.DataFrame(driver_rows)
    return features, stints

# ============================================================
# EXTRACCIÓN COMPLETA
# ============================================================
@st.cache_data(show_spinner=False)
def get_all_race_data(year, gp):
    setup_cache()

    # -----------------------------
    # 1. Prácticas
    # -----------------------------
    clean_practice = []
    for session_name in ["FP1", "FP2", "FP3"]:
        session = load_session_safe(year, gp, session_name)
        clean = prepare_clean_laps(session, session_name)
        if not clean.empty:
            clean_practice.append(clean)

    # Si no existe FP, Sprint puede servir como fuente secundaria de ritmo.
    if not clean_practice:
        session = load_session_safe(year, gp, "S")
        clean = prepare_clean_laps(session, "Sprint")
        if not clean.empty:
            clean_practice.append(clean)

    all_practice_laps = pd.concat(clean_practice, ignore_index=True) if clean_practice else pd.DataFrame()
    practice_features, stint_details = extract_long_run_features(all_practice_laps)

    # -----------------------------
    # 2. Qualifying principal
    # -----------------------------
    qualy_df = pd.DataFrame()
    q_session = load_session_safe(year, gp, "Q")
    if q_session is not None and hasattr(q_session, "results") and not q_session.results.empty:
        q_cols = [c for c in ["Abbreviation", "Position", "GridPosition"] if c in q_session.results.columns]
        qualy_df = q_session.results[q_cols].copy()
        qualy_df = qualy_df.rename(columns={"Abbreviation": "Driver", "Position": "q_position", "GridPosition": "grid_position"})

        try:
            q_laps = q_session.laps.pick_quicklaps().copy()
            q_laps["LapTime_sec"] = pd.to_timedelta(q_laps["LapTime"], errors="coerce").dt.total_seconds()
            q_laps = q_laps.dropna(subset=["LapTime_sec"])
            best_q = q_laps.groupby("Driver")["LapTime_sec"].min().reset_index()
            qualy_df = qualy_df.merge(best_q, on="Driver", how="left")
        except Exception:
            qualy_df["LapTime_sec"] = np.nan

        pole = qualy_df["LapTime_sec"].min() if "LapTime_sec" in qualy_df.columns else np.nan
        qualy_df["q_gap_sec"] = qualy_df["LapTime_sec"] - pole if pd.notna(pole) else np.nan

    # -----------------------------
    # 3. Sprint Qualifying
    # -----------------------------
    sq_df = pd.DataFrame()
    sq_session = load_session_safe(year, gp, "SQ")
    if sq_session is not None and hasattr(sq_session, "results") and not sq_session.results.empty:
        try:
            sq_df = sq_session.results[["Abbreviation", "Position"]].rename(
                columns={"Abbreviation": "Driver", "Position": "sq_position"}
            )
            sq_laps = sq_session.laps.pick_quicklaps().copy()
            sq_laps["LapTime_sec"] = pd.to_timedelta(sq_laps["LapTime"], errors="coerce").dt.total_seconds()
            sq_laps = sq_laps.dropna(subset=["LapTime_sec"])
            best_sq = sq_laps.groupby("Driver")["LapTime_sec"].min().reset_index()
            sq_df = sq_df.merge(best_sq, on="Driver", how="left")
        except Exception:
            sq_df["LapTime_sec"] = np.nan

    # -----------------------------
    # 4. Sprint race
    # -----------------------------
    sprint_results = pd.DataFrame()
    sprint_laps = pd.DataFrame()
    sprint_session = load_session_safe(year, gp, "S")
    if sprint_session is not None and hasattr(sprint_session, "results") and not sprint_session.results.empty:
        cols = [c for c in ["Abbreviation", "Position", "Points", "GridPosition"] if c in sprint_session.results.columns]
        sprint_results = sprint_session.results[cols].copy()
        sprint_results = sprint_results.rename(columns={
            "Abbreviation": "Driver",
            "Position": "s_result_pos",
            "Points": "s_points",
            "GridPosition": "s_grid_position",
        })
        try:
            sprint_laps = prepare_clean_laps(sprint_session, "Sprint")
        except Exception:
            sprint_laps = pd.DataFrame()

    # -----------------------------
    # 5. Colores y equipos
    # -----------------------------
    driver_colors = {}
    active_session = q_session
    if active_session is None or not hasattr(active_session, "results") or active_session.results.empty:
        for fallback in ["FP1", "FP2", "FP3", "SQ", "S"]:
            active_session = load_session_safe(year, gp, fallback)
            if active_session is not None and hasattr(active_session, "results") and not active_session.results.empty:
                break

    if active_session is not None and hasattr(active_session, "results"):
        team_counts = {}
        for _, row in active_session.results.iterrows():
            driver = row.get("Abbreviation")
            team = row.get("TeamName", "")
            if pd.isna(driver):
                continue
            driver = str(driver)
            team = str(team)
            color = get_team_color(team, driver_abbr=driver, session=active_session)
            count = team_counts.get(team, 0)
            driver_colors[driver] = color if count == 0 else darken_color(color)
            team_counts[team] = count + 1

    for driver, team in DRIVER_TEAM_2025.items():
        driver_colors.setdefault(driver, get_team_color(team, driver_abbr=driver))

    # -----------------------------
    # 6. Resultado real
    # -----------------------------
    actual_results = pd.DataFrame()
    actual_top_10 = []
    r_session = load_session_safe(year, gp, "R")
    if r_session is not None and hasattr(r_session, "results") and not r_session.results.empty:
        cols = [c for c in ["Abbreviation", "Position", "GridPosition", "Points"] if c in r_session.results.columns]
        actual_results = r_session.results[cols].copy().rename(columns={"Abbreviation": "Driver"})
        actual_results["Position"] = pd.to_numeric(actual_results["Position"], errors="coerce")
        actual_results = actual_results.dropna(subset=["Position"]).sort_values("Position")
        actual_top_10 = actual_results.head(10)["Driver"].tolist()

    return (
        practice_features,
        all_practice_laps,
        stint_details,
        qualy_df,
        sq_df,
        sprint_results,
        sprint_laps,
        driver_colors,
        actual_results,
        actual_top_10,
    )

# ============================================================
# PREDICCIÓN
# ============================================================
def build_prediction(practice_features, qualy_df, sprint_results, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    base_frames = []
    if not practice_features.empty:
        base_frames.append(practice_features.copy())
    if not qualy_df.empty:
        base_frames.append(qualy_df.copy())
    if not sprint_results.empty:
        base_frames.append(sprint_results[[c for c in ["Driver", "s_result_pos", "s_points"] if c in sprint_results.columns]].copy())

    if not base_frames:
        return pd.DataFrame()

    df = base_frames[0]
    for other in base_frames[1:]:
        df = df.merge(other, on="Driver", how="outer")

    # Grid real disponible; si no, fallback a qualy.
    if "grid_position" not in df.columns:
        df["grid_position"] = np.nan
    if "q_position" not in df.columns:
        df["q_position"] = np.nan
    df["start_position"] = df["grid_position"].combine_first(df["q_position"])

    # Score: todas las componentes 0..1, donde 0 = mejor.
    df["grid_component"] = normalize_rank(df["start_position"])
    df["qualy_component"] = normalize_rank(df["q_gap_sec"])
    df["long_run_component"] = normalize_rank(df["long_run_pace_gap_sec"])
    df["degradation_component"] = normalize_rank(df["degradation_slope"])
    df["consistency_component"] = normalize_rank(df["consistency_residual_std"])

    # Componentes disponibles: redistribuye el peso faltante, evitando penalizar
    # por ausencia de una sesión.
    component_map = {
        "grid": "grid_component",
        "qualifying": "qualy_component",
        "long_run": "long_run_component",
        "degradation": "degradation_component",
        "consistency": "consistency_component",
    }

    numerator = pd.Series(0.0, index=df.index)
    denominator = pd.Series(0.0, index=df.index)
    for weight_name, column in component_map.items():
        available = df[column].notna()
        numerator.loc[available] += weights[weight_name] * df.loc[available, column]
        denominator.loc[available] += weights[weight_name]

    df["pred_score"] = np.where(denominator > 0, numerator / denominator, np.nan)

    # Sprint es información adicional, pero no entra por defecto en el score
    # principal para no doble-contar qualifying/sprint position sin calibración.
    df["sprint_note"] = df.get("s_result_pos", pd.Series(index=df.index, dtype=float))

    df = df.sort_values(["pred_score", "start_position", "q_gap_sec"], na_position="last").reset_index(drop=True)
    df["pred_position"] = np.arange(1, len(df) + 1)
    return df

# ============================================================
# EVALUACIÓN
# ============================================================
def evaluate_prediction(prediction_df, actual_results):
    if prediction_df.empty or actual_results.empty:
        return {}

    actual = actual_results[["Driver", "Position"]].copy()
    actual["Position"] = pd.to_numeric(actual["Position"], errors="coerce")
    pred = prediction_df[["Driver", "pred_position"]].copy()
    merged = pred.merge(actual, on="Driver", how="inner")
    merged = merged.dropna(subset=["Position", "pred_position"])
    if merged.empty:
        return {}

    merged["abs_error"] = (merged["pred_position"] - merged["Position"]).abs()
    exact = (merged["pred_position"] == merged["Position"]).mean()

    try:
        spearman = merged["pred_position"].corr(merged["Position"], method="spearman")
    except Exception:
        spearman = np.nan

    pred_top10 = set(prediction_df.head(10)["Driver"])
    actual_top10 = set(actual_results.head(10)["Driver"])
    top10_hits = len(pred_top10 & actual_top10)

    pred_top3 = set(prediction_df.head(3)["Driver"])
    actual_top3 = set(actual_results.head(3)["Driver"])
    top3_hits = len(pred_top3 & actual_top3)

    f1_points = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
    actual_lookup = dict(zip(actual_results["Driver"].head(10), actual_results["Position"].head(10)))
    pred_top10_rows = prediction_df.head(10)
    f1_score = 0
    for idx, row in pred_top10_rows.iterrows():
        p = int(row["pred_position"])
        d = row["Driver"]
        if d in actual_lookup and int(actual_lookup[d]) == p:
            f1_score += f1_points[p - 1] if p <= 10 else 0

    return {
        "MAE": float(merged["abs_error"].mean()),
        "MedAE": float(merged["abs_error"].median()),
        "Exact Position %": float(exact * 100),
        "Spearman": float(spearman) if pd.notna(spearman) else np.nan,
        "Top 10 hits": int(top10_hits),
        "Top 3 hits": int(top3_hits),
        "F1 exact-position points": int(f1_score),
        "matched_drivers": int(len(merged)),
    }

# ============================================================
# VISUALIZACIONES
# ============================================================
def practice_boxplot(df, title, driver_colors):
    if df.empty:
        return None
    fig = px.box(
        df,
        x="Driver",
        y="LapTime_sec",
        color="Driver",
        title=title,
        labels={"LapTime_sec": "Tiempo por vuelta (s)"},
        color_discrete_map=driver_colors,
    )
    fig.update_layout(showlegend=False)
    return fig


def prediction_scatter(prediction_df, actual_results):
    if prediction_df.empty or actual_results.empty:
        return None
    df = prediction_df[["Driver", "pred_position"]].merge(
        actual_results[["Driver", "Position"]], on="Driver", how="inner"
    )
    df = df.rename(columns={"pred_position": "Predicción", "Position": "Realidad"})
    fig = px.scatter(
        df,
        x="Predicción",
        y="Realidad",
        text="Driver",
        title="Posición predicha vs posición real",
    )
    max_pos = int(max(df["Predicción"].max(), df["Realidad"].max()))
    fig.add_trace(go.Scatter(
        x=[1, max_pos], y=[1, max_pos], mode="lines",
        name="Predicción perfecta",
        line=dict(dash="dash"),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(autorange="reversed")
    fig.update_traces(textposition="top center")
    return fig

# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
def main():
    st.set_page_config(
        page_title="F1 Analytics & Prediction Engine",
        layout="wide",
        page_icon="🏁",
    )

    st.title("🏁 F1 ANALYTICS & PREDICTION ENGINE")
    st.markdown(
        "Modelo heurístico mejorado con **stints, TyreLife, compuesto, ritmo de tanda larga, "
        "degradación intra-stint, consistencia residual y posición de salida**."
    )

    st.sidebar.header("Configuración del evento")
    year = st.sidebar.number_input("Temporada", min_value=2018, max_value=2026, value=2025, step=1)
    available_gps = get_events_for_year(year)

    default_idx = 0
    for idx, event_name in enumerate(available_gps):
        if "Italian" in event_name or "Monza" in event_name:
            default_idx = idx
            break

    gp = st.sidebar.selectbox("Gran Premio", available_gps, index=default_idx)

    st.sidebar.markdown("### Pesos del modelo")
    w_grid = st.sidebar.slider("Grid", 0.00, 1.00, DEFAULT_WEIGHTS["grid"], 0.05)
    w_q = st.sidebar.slider("Qualifying gap", 0.00, 1.00, DEFAULT_WEIGHTS["qualifying"], 0.05)
    w_lr = st.sidebar.slider("Long-run pace", 0.00, 1.00, DEFAULT_WEIGHTS["long_run"], 0.05)
    w_deg = st.sidebar.slider("Degradación", 0.00, 1.00, DEFAULT_WEIGHTS["degradation"], 0.05)
    w_cons = st.sidebar.slider("Consistencia", 0.00, 1.00, DEFAULT_WEIGHTS["consistency"], 0.05)

    weight_sum = w_grid + w_q + w_lr + w_deg + w_cons
    if weight_sum <= 0:
        st.sidebar.error("La suma de pesos debe ser mayor que 0.")
        return

    weights = {
        "grid": w_grid / weight_sum,
        "qualifying": w_q / weight_sum,
        "long_run": w_lr / weight_sum,
        "degradation": w_deg / weight_sum,
        "consistency": w_cons / weight_sum,
    }

    execute = st.sidebar.button("Iniciar análisis", use_container_width=True)

    if execute or st.session_state.get("f1_analyzed", False):
        st.session_state["f1_analyzed"] = True

        with st.spinner(f"Cargando {gp} ({year}) desde FastF1..."):
            (
                practice_features,
                all_practice_laps,
                stint_details,
                qualy_df,
                sq_df,
                sprint_results,
                sprint_laps,
                driver_colors,
                actual_results,
                actual_top_10,
            ) = get_all_race_data(year, gp)

        prediction_df = build_prediction(practice_features, qualy_df, sprint_results, weights)
        metrics = evaluate_prediction(prediction_df, actual_results)

        if prediction_df.empty:
            st.error("No hubo datos suficientes para generar una predicción.")
            return

        st.caption(
            "Los pesos son configurables. El sistema redistribuye automáticamente el peso disponible "
            "cuando una sesión o métrica no existe. No son pesos calibrados estadísticamente hasta ejecutar backtesting."
        )

        tab_pred, tab_long, tab_qualy, tab_deg, tab_data = st.tabs([
            "🔮 Predicción vs realidad",
            "🏎️ Ritmo de tandas largas",
            "⏱️ Clasificación y Sprint",
            "📈 Degradación y consistencia",
            "📋 Datos del modelo",
        ])

        # ----------------------------------------------------
        # TAB PREDICCIÓN
        # ----------------------------------------------------
        with tab_pred:
            st.subheader("Ranking predictivo")
            top10 = prediction_df.head(10).copy()
            display_cols = [
                "pred_position", "Driver", "start_position", "q_gap_sec",
                "long_run_pace_gap_sec", "degradation_slope",
                "consistency_residual_std", "pred_score",
            ]
            display_cols = [c for c in display_cols if c in top10.columns]
            st.dataframe(top10[display_cols], use_container_width=True, hide_index=True)

            if metrics:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("MAE", f"{metrics['MAE']:.2f}")
                c2.metric("MedAE", f"{metrics['MedAE']:.2f}")
                c3.metric("Spearman", f"{metrics['Spearman']:.3f}")
                c4.metric("Top-10 hits", f"{metrics['Top 10 hits']}/10")
                c5.metric("Top-3 hits", f"{metrics['Top 3 hits']}/3")

                st.metric(
                    "Exact-position F1 points",
                    f"{metrics['F1 exact-position points']} / 101",
                )

                fig = prediction_scatter(prediction_df, actual_results)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("La carrera real todavía no permite calcular métricas de evaluación.")

        # ----------------------------------------------------
        # TAB LONG RUN
        # ----------------------------------------------------
        with tab_long:
            st.subheader("Ritmo de tandas largas por compuesto")
            if not stint_details.empty:
                fig = px.scatter(
                    stint_details,
                    x="Compound",
                    y="pace_gap_sec",
                    color="Driver",
                    size="Stint_Laps",
                    hover_data=["Session_Type", "Stint", "pace_median", "Stint_Laps"],
                    color_discrete_map=driver_colors,
                    title="Gap al mejor stint del mismo contexto (menor = mejor)",
                )
                fig.update_yaxes(title="Gap de ritmo (s/vuelta)")
                st.plotly_chart(fig, use_container_width=True)

                long_cols = [
                    "Driver", "long_run_pace_gap_sec", "long_run_pace_raw_sec",
                    "long_run_stints", "long_run_laps",
                ]
                st.dataframe(
                    prediction_df[long_cols].sort_values("long_run_pace_gap_sec"),
                    use_container_width=True,
                    hide_index=True,
                )
            elif not all_practice_laps.empty:
                st.info("Hay vueltas de práctica, pero no suficientes tandas de al menos 5 vueltas para el análisis de long runs.")
            else:
                st.info("No hay vueltas limpias disponibles.")

            if not all_practice_laps.empty:
                st.markdown("---")
                # Boxplot descriptivo de las vueltas limpias; no se usa directamente en la predicción.
                fig_box = practice_boxplot(all_practice_laps, "Distribución descriptiva de vueltas limpias", driver_colors)
                if fig_box is not None:
                    st.plotly_chart(fig_box, use_container_width=True)

        # ----------------------------------------------------
        # TAB QUALY / SPRINT
        # ----------------------------------------------------
        with tab_qualy:
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Qualifying")
                if not qualy_df.empty:
                    qplot = qualy_df.dropna(subset=["LapTime_sec"]).sort_values("LapTime_sec")
                    fig_q = px.bar(
                        qplot,
                        x="Driver",
                        y="LapTime_sec",
                        color="Driver",
                        title="Mejor vuelta de Qualy",
                        labels={"LapTime_sec": "Segundos"},
                        color_discrete_map=driver_colors,
                    )
                    st.plotly_chart(fig_q, use_container_width=True)
                    st.dataframe(
                        qualy_df.sort_values("q_position"),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No hay datos de Qualy.")

            with c2:
                st.subheader("Sprint")
                if not sprint_results.empty:
                    st.dataframe(sprint_results, use_container_width=True, hide_index=True)
                else:
                    st.info("No hubo Sprint o no existen datos disponibles.")

        # ----------------------------------------------------
        # TAB DEGRADACIÓN
        # ----------------------------------------------------
        with tab_deg:
            st.subheader("Degradación intra-stint")
            if not stint_details.empty:
                deg = stint_details.dropna(subset=["degradation_slope"]).copy()
                if not deg.empty:
                    fig_deg = px.scatter(
                        deg,
                        x="Stint_Laps",
                        y="degradation_slope",
                        color="Driver",
                        size="Stint_Laps",
                        hover_data=["Session_Type", "Compound", "Stint", "slope_axis", "pace_median"],
                        color_discrete_map=driver_colors,
                        title="Pendiente de tiempo por vuelta dentro de cada stint",
                        labels={
                            "Stint_Laps": "Vueltas analizadas en el stint",
                            "degradation_slope": "s/vuelta",
                        },
                    )
                    st.plotly_chart(fig_deg, use_container_width=True)

                fig_rank = px.bar(
                    prediction_df.sort_values("degradation_slope"),
                    x="Driver",
                    y="degradation_slope",
                    color="Driver",
                    title="Degradación agregada por piloto (menor = mejor)",
                    labels={"degradation_slope": "s/vuelta sobre TyreLife"},
                    color_discrete_map=driver_colors,
                )
                fig_rank.add_hline(y=0, line_dash="dash", opacity=0.5)
                st.plotly_chart(fig_rank, use_container_width=True)

                st.subheader("Consistencia alrededor de la tendencia")
                cons = prediction_df.sort_values("consistency_residual_std")
                fig_cons = px.bar(
                    cons,
                    x="Driver",
                    y="consistency_residual_std",
                    color="Driver",
                    title="STD de residuos del stint (menor = más consistente)",
                    labels={"consistency_residual_std": "Desviación residual (s)"},
                    color_discrete_map=driver_colors,
                )
                st.plotly_chart(fig_cons, use_container_width=True)
            else:
                st.info("No hay suficientes stints para estimar degradación y consistencia.")

        # ----------------------------------------------------
        # TAB DATA
        # ----------------------------------------------------
        with tab_data:
            st.subheader("Features usadas por el modelo")
            feature_cols = [
                "Driver", "pred_position", "pred_score",
                "start_position", "q_position", "q_gap_sec",
                "long_run_pace_gap_sec", "long_run_pace_raw_sec",
                "degradation_slope", "consistency_residual_std",
                "long_run_stints", "long_run_laps",
                "s_result_pos", "s_points",
            ]
            feature_cols = [c for c in feature_cols if c in prediction_df.columns]
            st.dataframe(prediction_df[feature_cols], use_container_width=True, hide_index=True)

            if not stint_details.empty:
                st.markdown("---")
                st.subheader("Stints detectados")
                st.dataframe(stint_details.sort_values(["Driver", "Session_Type", "Stint"]), use_container_width=True, hide_index=True)

            if not actual_results.empty:
                st.markdown("---")
                st.subheader("Resultado real")
                st.dataframe(actual_results, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
