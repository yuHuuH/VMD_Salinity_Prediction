from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import numpy as np
import pandas as pd
import streamlit as st
import folium
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from salinity_pipeline import (
    DYNAMIC_COLS,
    STATIC_COLS,
    FEATURES_REQUIRED,
    load_lstm_artifacts,
    InputDataError,
    build_model_frame_from_csv,
    canonical_history,
    discover_inventory,
    evaluate_against_recorded,
    make_results_gdf,
    predict_lstm,
    province_summary,
    read_boundary,
    read_csv_flexible,
    save_outputs,
    training_range_report,
    valid_year_table,
)



def _looks_like_project_root(path: Path) -> bool:
    """Detect a folder that contains the demo input folders.

    This lets the app work in normal Python mode, PyInstaller onedir mode,
    and PyInstaller onefile mode. When packaged, users can also place
    data/, boundary/, and trained_models/ next to the .exe.
    """
    return any((path / name).exists() for name in ("data", "boundary", "trained_models"))


def runtime_root() -> Path:
    app_dir = Path(__file__).resolve().parent

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if _looks_like_project_root(exe_dir):
            return exe_dir

        meipass = Path(getattr(sys, "_MEIPASS", app_dir))
        if _looks_like_project_root(meipass):
            return meipass

        return exe_dir

    return app_dir


ROOT = runtime_root()

st.set_page_config(page_title="Mekong Salinity — CSV Demo", page_icon="🌊", layout="wide")

# Page spacing: keep the large title below Streamlit's top toolbar,
# while keeping the map/report layout compact after the header.
st.markdown(
    """
    <style>
    .main .block-container,
    .block-container {
        padding-top: 4.75rem !important;
        padding-bottom: 0.8rem !important;
    }

    h1 {
        margin-top: 0 !important;
        padding-top: 0 !important;
        line-height: 1.15 !important;
    }

    div[data-testid="stVerticalBlock"] { gap: 0.35rem; }
    div[data-testid="column"] { gap: 0.35rem; }
    iframe[title="streamlit_folium.st_folium"] { display: block; margin-bottom: 0 !important; }
    div.element-container:has(iframe[title="streamlit_folium.st_folium"]) { margin-bottom: 0 !important; padding-bottom: 0 !important; }
    div[data-testid="stHorizontalBlock"] { gap: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

MAP_HEIGHT = 650
TABLE_HEIGHT = 650


@st.cache_data(show_spinner=False)
def load_history_cached(path: str, modified: float) -> pd.DataFrame:
    return canonical_history(read_csv_flexible(path))


@st.cache_data(show_spinner=False)
def load_boundary_cached(path: str, modified: float) -> gpd.GeoDataFrame:
    return read_boundary(path)


def file_stamp(path: Path) -> float:
    return path.stat().st_mtime


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def format_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


def selection_index(paths: list[Path], preferred_tokens: list[str]) -> int:
    for index, path in enumerate(paths):
        name = path.name.casefold()
        if all(token in name for token in preferred_tokens):
            return index
    return 0


def draw_map(
    results: gpd.GeoDataFrame,
    field: str,
    title: str,
    target_year: int,
    fixed_vmin: float | None = None,
    fixed_vmax: float | None = None,
):
    values = pd.to_numeric(results[field], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        raise InputDataError(f"No numeric values exist for map field: {field}")

    fig, ax = plt.subplots(figsize=(13, 11))
    if field in {"absolute_change", "percent_change"}:
        bound = max(float(np.nanpercentile(np.abs(values), 98)), 0.01)
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
        cmap = "RdBu_r"
    else:
        if fixed_vmin is not None and fixed_vmax is not None:
            norm = Normalize(vmin=float(fixed_vmin), vmax=float(fixed_vmax))
        else:
            norm = None
        cmap = "viridis"

    results.plot(
        column=field,
        cmap=cmap,
        norm=norm,
        linewidth=0.18,
        edgecolor="#ffffff",
        legend=True,
        legend_kwds={"shrink": 0.62, "label": title},
        ax=ax,
    )
    ax.set_title(f"Mekong Delta — {title}\nYear: {target_year}", fontsize=15, fontweight="bold", pad=14)
    ax.set_axis_off()
    fig.tight_layout()
    return fig



def _format_float(value: object, decimals: int = 2, signed: bool = False, suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        number = float(value)
        sign = "+" if signed else ""
        return f"{number:{sign}.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


def salinity_range_from_baseline(results: gpd.GeoDataFrame) -> tuple[float, float]:
    """Return the min/max salinity range from the baseline column only.

    This lets the predicted salinity map and selected report plots use the same
    visual scale as the baseline. It changes only the visualization range, not
    the model output values.
    """
    values = pd.to_numeric(results["Salinity_Last_Year"], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0.0, 1.0
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax




def salinity_range_from_columns(results: gpd.GeoDataFrame, columns: list[str]) -> tuple[float, float]:
    """Return one shared min/max range across multiple salinity columns.

    This is used for side-by-side baseline vs predicted maps so both maps use
    the same color meaning. It changes only the visualization scale.
    """
    values_list = []
    for column in columns:
        if column in results.columns:
            values = pd.to_numeric(results[column], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            if not values.empty:
                values_list.append(values)
    if not values_list:
        return 0.0, 1.0
    all_values = pd.concat(values_list, ignore_index=True)
    vmin = float(all_values.min())
    vmax = float(all_values.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def explain_scale_choice(scale_mode: str, field: str, fixed_range: tuple[float, float] | None) -> None:
    if scale_mode == "Match baseline salinity range" and field in {"Predicted_Salinity", "Salinity_Last_Year"} and fixed_range is not None:
        st.caption(
            f"Visualization scale: using baseline salinity range "
            f"{fixed_range[0]:.2f}–{fixed_range[1]:.2f} g/L. "
            "Prediction values are unchanged."
        )


def build_interactive_map(
    results: gpd.GeoDataFrame,
    field: str,
    title: str,
    target_year: int,
    fixed_vmin: float | None = None,
    fixed_vmax: float | None = None,
):
    """
    Folium map with hover tooltip, click popup, and highlighted commune boundary.
    The static matplotlib map is still used only for PNG export.
    """
    values = pd.to_numeric(results[field], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise InputDataError(f"No numeric values exist for map field: {field}")

    map_gdf = results.to_crs(epsg=4326).copy()

    # Simplify only the web copy so hover/click stays responsive.
    # The downloaded/static outputs still use the original geometry.
    try:
        map_gdf["geometry"] = map_gdf.geometry.simplify(0.00035, preserve_topology=True)
    except Exception:
        pass

    map_gdf["Baseline"] = map_gdf["Salinity_Last_Year"].map(lambda v: _format_float(v, 2, False, " g/L"))
    map_gdf["Predicted"] = map_gdf["Predicted_Salinity"].map(lambda v: _format_float(v, 2, False, " g/L"))
    map_gdf["Change"] = map_gdf["absolute_change"].map(lambda v: _format_float(v, 2, True, " g/L"))
    map_gdf["ChangePercent"] = map_gdf["percent_change"].map(lambda v: _format_float(v, 1, True, "%"))
    map_gdf["TrendIcon"] = np.where(
        map_gdf["absolute_change"] > 0.01,
        "Increase ↑",
        np.where(map_gdf["absolute_change"] < -0.01, "Decrease ↓", "Stable →"),
    )

    # Keep the GeoJSON small and JSON-safe.
    keep_columns = [
        "Province", "District", "Commune",
        "Baseline", "Predicted", "Change", "ChangePercent", "TrendIcon",
        "Salinity_Last_Year", "Predicted_Salinity", "absolute_change", "percent_change", field,
        "geometry",
    ]
    keep_columns = list(dict.fromkeys([col for col in keep_columns if col in map_gdf.columns]))
    map_gdf = map_gdf[keep_columns].copy()
    map_gdf = map_gdf.replace([np.inf, -np.inf], np.nan)
    for col in map_gdf.columns:
        if col != "geometry":
            map_gdf[col] = map_gdf[col].where(pd.notna(map_gdf[col]), None)

    minx, miny, maxx, maxy = map_gdf.total_bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]

    m = folium.Map(
        location=center,
        zoom_start=8,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )

    if field in {"absolute_change", "percent_change"}:
        bound = max(float(np.nanpercentile(np.abs(values), 98)), 0.01)
        colormap = LinearColormap(
            colors=["#2166ac", "#f7f7f7", "#b2182b"],
            vmin=-bound,
            vmax=bound,
        )
    else:
        if fixed_vmin is not None and fixed_vmax is not None:
            vmin = float(fixed_vmin)
            vmax = float(fixed_vmax)
        else:
            vmin = float(values.min())
            vmax = float(values.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        colormap = LinearColormap(
            colors=["#440154", "#21918c", "#fde725"],
            vmin=vmin,
            vmax=vmax,
        )

    colormap.caption = title

    # Compatibility fix for streamlit-folium.
    # Some branca LinearColormap versions do not expose default_css/default_js,
    # but streamlit-folium expects every map child to have them.
    if not hasattr(colormap, "default_css"):
        colormap.default_css = []
    if not hasattr(colormap, "default_js"):
        colormap.default_js = []

    def style_function(feature):
        raw_value = feature["properties"].get(field)
        try:
            numeric_value = float(raw_value)
            fill = colormap(numeric_value)
        except Exception:
            fill = "#cccccc"
        return {
            "fillColor": fill,
            "color": "#ffffff",
            "weight": 0.35,
            "fillOpacity": 0.72,
        }

    def highlight_function(feature):
        return {
            "fillColor": "#ffff99",
            "color": "#111111",
            "weight": 4,
            "fillOpacity": 0.92,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=["Province", "District", "Commune", "Baseline", "Predicted", "Change", "ChangePercent", "TrendIcon"],
        aliases=["Province", "District", "Commune", "Baseline", "Predicted", "Change", "Change (%)", "Trend"],
        localize=True,
        sticky=True,
        labels=True,
        style=(
            "background-color: white; color: #222; font-family: Arial; "
            "font-size: 12px; padding: 8px; border: 1px solid #999; border-radius: 4px;"
        ),
    )

    popup = folium.GeoJsonPopup(
        fields=["Province", "District", "Commune", "Baseline", "Predicted", "Change", "ChangePercent", "TrendIcon"],
        aliases=["Province", "District", "Commune", "Baseline", "Predicted", "Change", "Change (%)", "Trend"],
        localize=True,
        labels=True,
        max_width=320,
    )

    geojson_data = map_gdf.to_json(drop_id=True)

    folium.GeoJson(
        data=geojson_data,
        name=f"Salinity {target_year}",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=tooltip,
        popup=popup,
    ).add_to(m)

    colormap.add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def selected_commune_card(map_event: dict | None) -> None:
    if not isinstance(map_event, dict):
        return
    active = map_event.get("last_active_drawing")
    if not active or not isinstance(active, dict):
        return
    props = active.get("properties") or {}
    if not props:
        return

    st.markdown("**Selected commune**")
    st.markdown(
        f"`{props.get('Province', '')} | {props.get('District', '')} | {props.get('Commune', '')}`"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline", props.get("Baseline", "N/A"))
    c2.metric("Predicted", props.get("Predicted", "N/A"))
    c3.metric("Change", props.get("ChangePercent", "N/A"))



def render_detailed_change_list(result_gdf: gpd.GeoDataFrame, summary: pd.DataFrame, result_year: int, *, expanded: bool = True) -> None:
    """Render the reusable detailed list panel for the interactive map section."""
    container = st.expander("Detailed change list", expanded=expanded) if expanded is not None else st.container()
    with container:
        list_level = st.radio(
            "List level",
            ["Commune details", "Province summary"],
            horizontal=True,
            key=f"list_level_{result_year}_{'expanded' if expanded else 'compact'}",
        )
        if list_level == "Commune details":
            table = make_compact_commune_list(result_gdf)
            province = st.selectbox(
                "Province filter",
                ["All provinces", *sorted(table["Province"].dropna().unique().tolist())],
                key=f"province_filter_{result_year}_{'expanded' if expanded else 'compact'}",
            )
            search_text = st.text_input(
                "Find province, district, commune",
                placeholder="Example: Tra Vinh",
                key=f"search_text_{result_year}_{'expanded' if expanded else 'compact'}",
            ).strip()
            if province != "All provinces":
                table = table.loc[table["Province"] == province].copy()
            if search_text:
                search_frame = table[["Province", "District", "Commune"]].fillna("").astype(str)
                matched = search_frame.apply(lambda column: column.str.contains(search_text, case=False, regex=False)).any(axis=1)
                table = table.loc[matched].copy()
            display_columns = ["Province", "District", "Commune", "Salinity", "Δ g/L", "Δ%", "Trend"]
            filename = f"commune_change_list_{result_year}.csv"
        else:
            table = make_compact_province_list(summary)
            display_columns = ["Province", "Salinity", "Δ g/L", "Δ%", "Trend"]
            filename = f"province_summary_{result_year}.csv"

        sort_option = st.selectbox(
            "Sort list by",
            ["Largest increase (%)", "Largest decrease (%)", "Largest increase (g/L)", "Largest decrease (g/L)", "Province A→Z", "Province Z→A"],
            index=0,
            key=f"sort_option_{result_year}_{'expanded' if expanded else 'compact'}",
        )
        table = sort_change_table(table, sort_option)
        st.dataframe(table[display_columns], hide_index=True, use_container_width=True, height=TABLE_HEIGHT)
        st.download_button(
            "Download displayed list CSV",
            csv_bytes(table),
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
            key=f"download_list_{result_year}_{'expanded' if expanded else 'compact'}",
        )


def draw_province_change_bar(summary: pd.DataFrame, sort_by: str):
    plot_df = summary.copy()
    plot_df = plot_df.sort_values(sort_by, ascending=(sort_by == "Province"), kind="stable")
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(plot_df))))
    ax.barh(plot_df["Province"], plot_df["percent_change"])
    ax.set_title("Province change ranking (%)", fontweight="bold")
    ax.set_xlabel("Percent change (%)")
    ax.set_ylabel("Province")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_baseline_vs_predicted(summary: pd.DataFrame, match_baseline_range: bool = False):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(summary["baseline_salinity"], summary["predicted_salinity"])
    if match_baseline_range:
        minimum = float(summary["baseline_salinity"].min())
        maximum = float(summary["baseline_salinity"].max())
    else:
        minimum = float(min(summary["baseline_salinity"].min(), summary["predicted_salinity"].min()))
        maximum = float(max(summary["baseline_salinity"].max(), summary["predicted_salinity"].max()))
    if np.isclose(minimum, maximum):
        maximum = minimum + 1e-6
    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    ax.set_xlim(minimum, maximum)
    ax.set_ylim(minimum, maximum)
    ax.set_title("Province mean: baseline vs predicted", fontweight="bold")
    ax.set_xlabel("Baseline salinity (g/L)")
    ax.set_ylabel("Predicted salinity (g/L)")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_commune_change_hist(results: gpd.GeoDataFrame):
    values = pd.to_numeric(results["percent_change"], errors="coerce")
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(values, bins=30)
    ax.axvline(0, linestyle="--")
    ax.set_title("Commune percent change distribution", fontweight="bold")
    ax.set_xlabel("Percent change (%)")
    ax.set_ylabel("Number of communes")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig



def add_actual_comparison_columns(results: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add actual CSV change/error columns for report plots when Actual_Salinity exists."""
    out = results.copy()
    if "Actual_Salinity" in out.columns:
        out["actual_absolute_change"] = out["Actual_Salinity"] - out["Salinity_Last_Year"]
        out["actual_percent_change"] = np.where(
            np.abs(out["Salinity_Last_Year"]) > 1e-6,
            out["actual_absolute_change"] / out["Salinity_Last_Year"] * 100.0,
            np.nan,
        )
        out["prediction_error"] = out["Predicted_Salinity"] - out["Actual_Salinity"]
        out["absolute_error"] = np.abs(out["prediction_error"])
        out["actual_trend"] = np.select(
            [out["actual_absolute_change"] > 0.01, out["actual_absolute_change"] < -0.01],
            ["Increase", "Decrease"],
            default="Stable",
        )
    else:
        for col in [
            "actual_absolute_change", "actual_percent_change", "prediction_error",
            "absolute_error", "actual_trend",
        ]:
            out[col] = np.nan
    return out


def _safe_report_frame(results: gpd.GeoDataFrame) -> pd.DataFrame:
    return pd.DataFrame(results.drop(columns="geometry", errors="ignore")).copy()


def _short_label(row: pd.Series) -> str:
    province = str(row.get("Province", ""))
    commune = str(row.get("Commune", ""))
    district = str(row.get("District", ""))
    label = f"{commune} — {district}, {province}"
    return label if len(label) <= 55 else label[:52] + "..."


def draw_commune_baseline_vs_predicted(results: gpd.GeoDataFrame, match_baseline_range: bool = False):
    data = _safe_report_frame(results).dropna(subset=["Salinity_Last_Year", "Predicted_Salinity"])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(data["Salinity_Last_Year"], data["Predicted_Salinity"], s=12, alpha=0.55)
    if match_baseline_range:
        minimum = float(data["Salinity_Last_Year"].min())
        maximum = float(data["Salinity_Last_Year"].max())
    else:
        minimum = float(min(data["Salinity_Last_Year"].min(), data["Predicted_Salinity"].min()))
        maximum = float(max(data["Salinity_Last_Year"].max(), data["Predicted_Salinity"].max()))
    if np.isclose(minimum, maximum):
        maximum = minimum + 1e-6
    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    ax.set_xlim(minimum, maximum)
    ax.set_ylim(minimum, maximum)
    ax.set_title("Commune-level baseline vs predicted salinity", fontweight="bold")
    ax.set_xlabel("Baseline salinity (g/L)")
    ax.set_ylabel("Predicted salinity (g/L)")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_salinity_distribution(results: gpd.GeoDataFrame, match_baseline_range: bool = False):
    data = _safe_report_frame(results)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    columns = [
        ("Salinity_Last_Year", "Baseline"),
        ("Predicted_Salinity", "Predicted"),
    ]
    if "Actual_Salinity" in data.columns and data["Actual_Salinity"].notna().any():
        columns.append(("Actual_Salinity", "Ground Truth"))

    hist_bins = 35
    if match_baseline_range:
        baseline_values = pd.to_numeric(data["Salinity_Last_Year"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not baseline_values.empty:
            x_min = float(baseline_values.min())
            x_max = float(baseline_values.max())
            if np.isclose(x_min, x_max):
                x_max = x_min + 1e-6
            hist_bins = np.linspace(x_min, x_max, 36)
            ax.set_xlim(x_min, x_max)

    for col, label in columns:
        values = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            ax.hist(values, bins=hist_bins, alpha=0.45, label=label)
    ax.set_title("Commune salinity distribution", fontweight="bold")
    ax.set_xlabel("Salinity (g/L)")
    ax.set_ylabel("Number of communes")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_top_commune_change_bar(results: gpd.GeoDataFrame, n: int = 15, largest: bool = True):
    data = _safe_report_frame(results).dropna(subset=["percent_change"]).copy()
    data["label"] = data.apply(_short_label, axis=1)
    plot_df = data.sort_values("percent_change", ascending=not largest).head(n)
    plot_df = plot_df.sort_values("percent_change", ascending=True)

    title = f"Top {n} commune predicted increases (%)" if largest else f"Top {n} commune predicted decreases (%)"
    fig, ax = plt.subplots(figsize=(10, max(5.0, 0.38 * len(plot_df))))
    ax.barh(plot_df["label"], plot_df["percent_change"])
    ax.axvline(0, linestyle="--", linewidth=1.2)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted change (%)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_commune_absolute_change_ranking(results: gpd.GeoDataFrame, n: int = 15):
    data = _safe_report_frame(results).dropna(subset=["absolute_change"]).copy()
    data["label"] = data.apply(_short_label, axis=1)
    data["abs_rank_value"] = data["absolute_change"].abs()
    plot_df = data.sort_values("abs_rank_value", ascending=False).head(n)
    plot_df = plot_df.sort_values("absolute_change", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5.0, 0.38 * len(plot_df))))
    ax.barh(plot_df["label"], plot_df["absolute_change"])
    ax.axvline(0, linestyle="--", linewidth=1.2)
    ax.set_title(f"Top {n} communes by absolute predicted change", fontweight="bold")
    ax.set_xlabel("Predicted change (g/L)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_boxplot_change_by_province(results: gpd.GeoDataFrame):
    data = _safe_report_frame(results).dropna(subset=["Province", "percent_change"])
    provinces = sorted(data["Province"].dropna().unique().tolist())
    grouped = [data.loc[data["Province"] == province, "percent_change"].to_numpy() for province in provinces]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.boxplot(grouped, labels=provinces, showfliers=False)
    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.set_title("Distribution of commune predicted change by province", fontweight="bold")
    ax.set_ylabel("Predicted change (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_actual_vs_predicted_commune_scatter(results: gpd.GeoDataFrame):
    data = _safe_report_frame(results).dropna(subset=["Actual_Salinity", "Predicted_Salinity"])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(data["Actual_Salinity"], data["Predicted_Salinity"], s=13, alpha=0.55)
    minimum = float(min(data["Actual_Salinity"].min(), data["Predicted_Salinity"].min()))
    maximum = float(max(data["Actual_Salinity"].max(), data["Predicted_Salinity"].max()))
    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    mae = float(data["absolute_error"].mean()) if "absolute_error" in data else np.nan
    rmse = float(np.sqrt(np.mean(data["prediction_error"] ** 2))) if "prediction_error" in data else np.nan
    ax.set_title("Commune actual vs predicted salinity", fontweight="bold")
    ax.set_xlabel("Actual CSV salinity (g/L)")
    ax.set_ylabel("Predicted salinity (g/L)")
    ax.text(
        0.05, 0.95,
        f"MAE = {mae:.3f} g/L\nRMSE = {rmse:.3f} g/L",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_actual_vs_predicted_change_scatter(results: gpd.GeoDataFrame):
    data = _safe_report_frame(results).dropna(subset=["actual_percent_change", "percent_change"])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(data["actual_percent_change"], data["percent_change"], s=13, alpha=0.55)
    minimum = float(min(data["actual_percent_change"].min(), data["percent_change"].min()))
    maximum = float(max(data["actual_percent_change"].max(), data["percent_change"].max()))
    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    ax.axhline(0, linestyle=":", linewidth=1.0)
    ax.axvline(0, linestyle=":", linewidth=1.0)
    ax.set_title("Actual vs predicted commune salinity change", fontweight="bold")
    ax.set_xlabel("Actual CSV change (%)")
    ax.set_ylabel("Predicted change (%)")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_error_distribution(results: gpd.GeoDataFrame):
    data = _safe_report_frame(results).dropna(subset=["prediction_error"])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.hist(data["prediction_error"], bins=35, alpha=0.85)
    ax.axvline(0, linestyle="--", linewidth=1.4)
    ax.set_title("Commune prediction error distribution", fontweight="bold")
    ax.set_xlabel("Prediction error: predicted - actual (g/L)")
    ax.set_ylabel("Number of communes")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_province_error_summary(results: gpd.GeoDataFrame):
    data = _safe_report_frame(results).dropna(subset=["Province", "prediction_error", "absolute_error"])
    summary = data.groupby("Province", as_index=False).agg(
        mean_error=("prediction_error", "mean"),
        mae=("absolute_error", "mean"),
    )
    summary = summary.sort_values("mae", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5.0, 0.42 * len(summary))))
    y = np.arange(len(summary))
    width = 0.36
    ax.barh(y - width / 2, summary["mae"], height=width, label="MAE")
    ax.barh(y + width / 2, summary["mean_error"], height=width, label="Mean error")
    ax.axvline(0, linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["Province"])
    ax.set_title("Province-level model error summary", fontweight="bold")
    ax.set_xlabel("Error (g/L)")
    ax.legend()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_risk_category_bar(results: gpd.GeoDataFrame):
    data = _safe_report_frame(results).copy()
    bins = [-np.inf, 1, 4, 10, np.inf]
    labels = ["<1 g/L", "1–4 g/L", "4–10 g/L", ">10 g/L"]
    data["risk_category"] = pd.cut(data["Predicted_Salinity"], bins=bins, labels=labels)
    counts = data["risk_category"].value_counts().reindex(labels).fillna(0)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Predicted salinity risk categories", fontweight="bold")
    ax.set_xlabel("Predicted salinity category")
    ax.set_ylabel("Number of communes")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig



def _model_frame_signature(frame: pd.DataFrame) -> str:
    """Small signature for cached explainability outputs."""
    parts = [str(len(frame))]
    for column in ["Year", "Baseline_Year"]:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                parts.append(f"{column}:{int(values.min())}-{int(values.max())}")
    return "|".join(parts)


def _feature_columns_for_explainability() -> list[str]:
    dynamic_flat = [column for group in DYNAMIC_COLS for column in group]
    return [*STATIC_COLS, *dynamic_flat]


def _prepare_flat_features(frame: pd.DataFrame, sample_size: int, random_seed: int) -> pd.DataFrame:
    columns = _feature_columns_for_explainability()
    data = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().copy()
    if data.empty:
        raise InputDataError("No complete feature rows are available for SHAP explainability.")
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=random_seed)
    return data


def _predict_from_flat_features(x_flat: np.ndarray, model_path: str, pkl_path: str) -> np.ndarray:
    """Run ONNX LSTM from flattened [static + dynamic] feature rows."""
    feature_columns = _feature_columns_for_explainability()
    x_flat = np.asarray(x_flat, dtype=np.float32)
    if x_flat.ndim == 1:
        x_flat = x_flat.reshape(1, -1)
    if x_flat.shape[1] != len(feature_columns):
        raise ValueError(f"Expected {len(feature_columns)} features, received {x_flat.shape[1]}.")

    n_static = len(STATIC_COLS)
    x_static = x_flat[:, :n_static].astype(np.float32, copy=False)
    x_dynamic_flat = x_flat[:, n_static:].astype(np.float32, copy=False)
    x_dynamic = x_dynamic_flat.reshape(x_flat.shape[0], len(DYNAMIC_COLS), len(DYNAMIC_COLS[0]))

    session, package, dynamic_name, static_name = load_lstm_artifacts(model_path, pkl_path)
    x_dynamic_scaled = package["scaler_dyn"].transform(
        x_dynamic.reshape(-1, x_dynamic.shape[-1])
    ).reshape(x_dynamic.shape).astype(np.float32, copy=False)
    x_static_scaled = package["scaler_stat"].transform(x_static).astype(np.float32, copy=False)
    outputs = session.run(None, {dynamic_name: x_dynamic_scaled, static_name: x_static_scaled})
    return np.asarray(outputs[0]).reshape(-1).astype(float)


def _feature_label(feature: str) -> str:
    replacements = {
        "Salinity_Last_Year": "Baseline salinity",
        "NDWI_mean": "NDWI mean",
        "centroid_x": "Centroid X",
        "centroid_y": "Centroid Y",
        "rain_Sum_MuaMua": "Rainy-season rainfall",
        "t2m_Mean_MuaMua": "Rainy-season temperature",
        "evap_Sum_MuaMua": "Rainy-season evaporation",
        "runoff_Sum_MuaMua": "Rainy-season runoff",
        "soil_Mean_MuaMua": "Rainy-season soil water",
        "ONI_Rainy_LastYear": "Previous rainy-season ONI",
    }
    if feature in replacements:
        return replacements[feature]
    for tag in ["M11_prev", "M12_prev", "M1", "M2", "M3", "M4"]:
        suffix = f"_{tag}"
        if feature.endswith(suffix):
            base = feature[:-len(suffix)]
            return f"{base.replace('_', ' ').title()} ({tag})"
    return feature.replace("_", " ").title()


@st.cache_data(show_spinner=False)
def compute_kernel_shap_summary(
    frame_csv: str,
    model_path: str,
    pkl_path: str,
    sample_size: int,
    background_size: int,
    nsamples: int,
    random_seed: int,
) -> dict[str, Any]:
    """Compute Kernel SHAP on a small sample for the ONNX LSTM model."""
    try:
        import shap
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if getattr(sys, "frozen", False):
            raise InputDataError(
                "SHAP could not be loaded inside the compiled EXE. "
                "Rebuild the EXE with the updated build_exe.bat so PyInstaller bundles "
                "SHAP, Numba, llvmlite, SciPy, and related dependencies. "
                f"Import detail: {detail}"
            ) from exc
        raise InputDataError(
            "SHAP is not installed or failed to import in this Python environment. "
            "Run `python -m pip install shap==0.46.0` or reinstall requirements.txt. "
            f"Import detail: {detail}"
        ) from exc

    frame = pd.read_json(io.StringIO(frame_csv), orient="split")
    feature_columns = _feature_columns_for_explainability()
    sample = _prepare_flat_features(frame, sample_size=sample_size, random_seed=random_seed)

    # Background rows are the reference distribution used by Kernel SHAP.
    background_size = min(background_size, len(sample))
    background = sample.sample(n=background_size, random_state=random_seed + 7)

    def model_function(x: np.ndarray) -> np.ndarray:
        return _predict_from_flat_features(x, model_path=model_path, pkl_path=pkl_path)

    explainer = shap.KernelExplainer(model_function, background.to_numpy(dtype=np.float32))
    shap_values = explainer.shap_values(
        sample.to_numpy(dtype=np.float32),
        nsamples=nsamples,
        silent=True,
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    summary = pd.DataFrame({
        "feature": feature_columns,
        "feature_label": [_feature_label(column) for column in feature_columns],
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False, kind="stable")

    return {
        "summary": summary,
        "shap_values": shap_values,
        "sample_features": sample.reset_index(drop=True),
        "feature_columns": feature_columns,
    }


def draw_shap_importance_bar(shap_summary: pd.DataFrame, top_n: int = 20):
    plot_df = shap_summary.head(top_n).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5.2, 0.36 * len(plot_df))))
    ax.barh(plot_df["feature_label"], plot_df["mean_abs_shap"])
    ax.set_title("SHAP global feature importance", fontweight="bold")
    ax.set_xlabel("Mean |SHAP value| contribution to salinity prediction (g/L)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def draw_shap_beeswarm_like(shap_values: np.ndarray, sample_features: pd.DataFrame, shap_summary: pd.DataFrame, top_n: int = 15):
    top_features = shap_summary.head(top_n)["feature"].tolist()
    labels = shap_summary.head(top_n)["feature_label"].tolist()
    fig, ax = plt.subplots(figsize=(10, max(5.5, 0.42 * top_n)))

    rng = np.random.default_rng(42)
    for y_index, feature in enumerate(reversed(top_features)):
        feature_index = sample_features.columns.get_loc(feature)
        x_values = shap_values[:, feature_index]
        raw_feature = pd.to_numeric(sample_features[feature], errors="coerce").to_numpy(dtype=float)
        if np.nanmax(raw_feature) > np.nanmin(raw_feature):
            colors = (raw_feature - np.nanmin(raw_feature)) / (np.nanmax(raw_feature) - np.nanmin(raw_feature))
        else:
            colors = np.full_like(raw_feature, 0.5, dtype=float)
        y_jitter = rng.normal(loc=y_index, scale=0.055, size=len(x_values))
        scatter = ax.scatter(x_values, y_jitter, c=colors, s=18, alpha=0.72, cmap="coolwarm", edgecolors="none")

    ax.axvline(0, linestyle="--", linewidth=1.2)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel("SHAP value contribution to predicted salinity (g/L)")
    ax.set_title("SHAP summary plot", fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.78)
    cbar.set_label("Feature value: low → high")
    fig.tight_layout()
    return fig


def draw_shap_dependence(shap_values: np.ndarray, sample_features: pd.DataFrame, feature: str):
    feature_index = sample_features.columns.get_loc(feature)
    x_values = pd.to_numeric(sample_features[feature], errors="coerce").to_numpy(dtype=float)
    y_values = shap_values[:, feature_index]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.scatter(x_values, y_values, s=22, alpha=0.68)
    ax.axhline(0, linestyle="--", linewidth=1.1)
    ax.set_title(f"SHAP dependence — {_feature_label(feature)}", fontweight="bold")
    ax.set_xlabel(_feature_label(feature))
    ax.set_ylabel("SHAP value contribution (g/L)")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig

def make_compact_commune_list(results: gpd.GeoDataFrame) -> pd.DataFrame:
    compact = pd.DataFrame(results.drop(columns="geometry", errors="ignore")).copy()
    compact["Trend"] = np.where(
        compact["percent_change"] > 0.01,
        "🟢⬆",
        np.where(compact["percent_change"] < -0.01, "🔴⬇", "⚪→"),
    )
    compact["Salinity"] = compact.apply(
        lambda row: f"{row['Salinity_Last_Year']:.2f} → {row['Predicted_Salinity']:.2f}", axis=1
    )
    compact["Δ g/L"] = compact["absolute_change"].map(lambda value: f"{value:+.2f}" if pd.notna(value) else "N/A")
    compact["Δ%"] = compact["percent_change"].map(lambda value: f"{value:+.1f}%" if pd.notna(value) else "N/A")
    return compact[[
        "Province", "District", "Commune", "Salinity", "Δ g/L", "Δ%", "Trend",
        "percent_change", "absolute_change", "Salinity_Last_Year", "Predicted_Salinity",
    ]]


def make_compact_province_list(summary: pd.DataFrame) -> pd.DataFrame:
    compact = summary.copy()
    compact["Trend"] = np.where(
        compact["percent_change"] > 0.01,
        "🟢⬆",
        np.where(compact["percent_change"] < -0.01, "🔴⬇", "⚪→"),
    )
    compact["Salinity"] = compact.apply(
        lambda row: f"{row['baseline_salinity']:.2f} → {row['predicted_salinity']:.2f}", axis=1
    )
    compact["Δ g/L"] = compact["absolute_change"].map(lambda value: f"{value:+.2f}" if pd.notna(value) else "N/A")
    compact["Δ%"] = compact["percent_change"].map(lambda value: f"{value:+.1f}%" if pd.notna(value) else "N/A")
    return compact[[
        "Province", "Salinity", "Δ g/L", "Δ%", "Trend",
        "percent_change", "absolute_change", "baseline_salinity", "predicted_salinity",
    ]]


def sort_change_table(frame: pd.DataFrame, option: str) -> pd.DataFrame:
    if option == "Largest increase (%)":
        return frame.sort_values("percent_change", ascending=False, kind="stable")
    if option == "Largest decrease (%)":
        return frame.sort_values("percent_change", ascending=True, kind="stable")
    if option == "Largest increase (g/L)":
        return frame.sort_values("absolute_change", ascending=False, kind="stable")
    if option == "Largest decrease (g/L)":
        return frame.sort_values("absolute_change", ascending=True, kind="stable")
    if option == "Province A→Z":
        columns = [column for column in ["Province", "District", "Commune"] if column in frame.columns]
        return frame.sort_values(columns, ascending=True, kind="stable")
    if option == "Province Z→A":
        columns = [column for column in ["Province", "District", "Commune"] if column in frame.columns]
        return frame.sort_values(columns, ascending=False, kind="stable")
    return frame


def display_inventory(inventory) -> None:
    with st.expander("Detected local files", expanded=False):
        groups = [
            ("ONNX LSTM model", inventory.onnx_models),
            ("Scaler package (.pkl)", inventory.pkl_packages),
            ("Boundary", inventory.boundaries),
            ("Prepared Final CSV", inventory.history_csvs),
        ]
        columns = st.columns(2)
        for index, (label, paths) in enumerate(groups):
            with columns[index % 2]:
                st.markdown(f"**{label}**")
                if paths:
                    for path in paths:
                        st.code(format_path(path), language=None)
                else:
                    st.caption("Not found")


def prediction_signature(
    model_path: Path,
    pkl_path: Path,
    boundary_path: Path,
    history_path: Path,
    target_year: int,
) -> tuple:
    file_parts = tuple((str(path.resolve()), file_stamp(path)) for path in [model_path, pkl_path, boundary_path, history_path])
    return file_parts, int(target_year)



def keep_shap_section_open():
    """Keep Streamlit on the SHAP section after the SHAP button reruns the app."""
    st.session_state["main_section"] = "Research / report plots"
    st.session_state["report_section"] = "SHAP explainability"

def run_app() -> None:
    st.title("🌊 Mekong Delta Salinity Prediction")
    st.caption("CSV-only mode: selects prepared feature rows directly from VMD_Salinity_Dataset_IDW_500_Final.csv.")

    with st.sidebar:
        st.header("Project control")
        if st.button("Refresh local files", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Project folder\n`{ROOT}`")

    inventory = discover_inventory(ROOT)
    display_inventory(inventory)

    missing: list[str] = []
    if not inventory.onnx_models:
        missing.append("ONNX LSTM model (.onnx) in trained_models/")
    if not inventory.pkl_packages:
        missing.append("Scaler package (.pkl) in trained_models/")
    if not inventory.boundaries:
        missing.append("GADM level-3 boundary (.json/.geojson/.gpkg) in boundary/")
    if not inventory.history_csvs:
        missing.append("prepared VMD_Salinity_Dataset_IDW_500_Final.csv in data/")
    if missing:
        st.error("The demo cannot start yet. Add these items:\n\n- " + "\n- ".join(missing))
        return

    try:
        defaults = {
            "model": selection_index(inventory.onnx_models, ["lstm"]),
            "pkl": selection_index(inventory.pkl_packages, ["lstm"]),
            "boundary": selection_index(inventory.boundaries, ["gadm"]),
            "history": selection_index(inventory.history_csvs, ["final"]),
        }
        with st.sidebar:
            with st.expander("Advanced local-file selection", expanded=False):
                model_path = st.selectbox("ONNX LSTM model", inventory.onnx_models, index=defaults["model"], format_func=format_path)
                pkl_path = st.selectbox("Scaler package", inventory.pkl_packages, index=defaults["pkl"], format_func=format_path)
                boundary_path = st.selectbox("Boundary", inventory.boundaries, index=defaults["boundary"], format_func=format_path)
                history_path = st.selectbox("Prepared Final CSV", inventory.history_csvs, index=defaults["history"], format_func=format_path)
            if "model_path" not in locals():
                model_path = inventory.onnx_models[defaults["model"]]
                pkl_path = inventory.pkl_packages[defaults["pkl"]]
                boundary_path = inventory.boundaries[defaults["boundary"]]
                history_path = inventory.history_csvs[defaults["history"]]

        history = load_history_cached(str(history_path), file_stamp(history_path))
        status = valid_year_table(history)
    except InputDataError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.exception(exc)
        return

    valid_years = status.loc[status["status"] == "Valid", "target_year"].astype(int).tolist()
    if not valid_years:
        st.error("No selectable year contains all prepared model features and a previous salinity baseline.")
        st.dataframe(status, hide_index=True, use_container_width=True)
        return

    target_year = st.selectbox("Choose a year", valid_years, index=len(valid_years) - 1)
    with st.expander("CSV year availability", expanded=False):
        st.dataframe(status, hide_index=True, use_container_width=True)

    signature = prediction_signature(model_path, pkl_path, boundary_path, history_path, int(target_year))
    saved = st.session_state.get("prediction_bundle")
    matches = bool(saved and saved.get("signature") == signature)
    if saved and not matches:
        st.warning("Core prediction inputs changed. Press Run prediction / update results to calculate the selected year.")

    run = st.button("Run prediction / update results", type="primary", use_container_width=True)
    if run:
        try:
            with st.spinner("Preparing CSV features, scaling with saved objects, and running ONNX inference…"):
                communes = load_boundary_cached(str(boundary_path), file_stamp(boundary_path))
                frame, display_communes, baseline_years = build_model_frame_from_csv(
                    history=history,
                    communes=communes,
                    target_year=int(target_year),
                )
                predictions = predict_lstm(frame, model_path, pkl_path)
                result_gdf = make_results_gdf(display_communes, frame, predictions, int(target_year))
                summary = province_summary(pd.DataFrame(result_gdf.drop(columns="geometry")))
                ranges = training_range_report(history, frame)
                metadata: dict[str, Any] = {
                    "target_year": int(target_year),
                    "mode": "CSV-only prepared feature inference",
                    "onnx_model": model_path.name,
                    "scaler_package": pkl_path.name,
                    "boundary": boundary_path.name,
                    "history": history_path.name,
                    "baseline_years_used": baseline_years,
                    "dynamic_feature_order": [list(group) for group in DYNAMIC_COLS],
                    "static_feature_order": STATIC_COLS,
                }
                output_dir = save_outputs(ROOT, result_gdf, summary, metadata)

            st.session_state["prediction_bundle"] = {
                "signature": signature,
                "target_year": int(target_year),
                "result_gdf": result_gdf,
                "summary": summary,
                "ranges": ranges,
                "baseline_years": baseline_years,
                "output_dir": output_dir,
                "model_frame": frame,
                "model_path": str(model_path),
                "pkl_path": str(pkl_path),
            }
            saved = st.session_state["prediction_bundle"]
            matches = True
        except InputDataError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.exception(exc)
            return

    if not matches:
        st.info("Choose a year and press Run prediction / update results once. Map controls, sorting, filters, and charts will then reuse the saved result.")
        return

    result_gdf: gpd.GeoDataFrame = add_actual_comparison_columns(saved["result_gdf"])
    summary: pd.DataFrame = saved["summary"]
    ranges: pd.DataFrame = saved["ranges"]
    output_dir: Path = saved["output_dir"]
    model_frame: pd.DataFrame | None = saved.get("model_frame")
    saved_model_path: str = saved.get("model_path", str(model_path))
    saved_pkl_path: str = saved.get("pkl_path", str(pkl_path))
    result_year = int(saved["target_year"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Communes predicted", f"{len(result_gdf):,}")
    c2.metric("Mean predicted salinity", f"{result_gdf['Predicted_Salinity'].mean():.2f} g/L")
    c3.metric("Mean change", f"{result_gdf['absolute_change'].mean():+.2f} g/L")
    c4.metric("Province trends", f"{int((summary['trend'] == 'Increase').sum())} ↑ / {int((summary['trend'] == 'Decrease').sum())} ↓")

    if st.checkbox("Compare with recorded Salinity_max from CSV", value=False):
        try:
            metrics = evaluate_against_recorded(result_gdf)
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Compared communes", f"{metrics['n']:,}")
            e2.metric("MAE", f"{metrics['MAE']:.3f} g/L")
            e3.metric("RMSE", f"{metrics['RMSE']:.3f} g/L")
            e4.metric("R²", f"{metrics['R2']:.4f}" if np.isfinite(metrics["R2"]) else "N/A")
        except InputDataError as exc:
            st.warning(str(exc))

    st.divider()

    main_section = st.radio(
        "Main section",
        ["Interactive map", "Research / report plots", "Report tables / exports"],
        horizontal=True,
        key="main_section",
    )

    if main_section == "Interactive map":
        st.subheader("Interactive map view")

        display_mode = st.radio(
            "Map display",
            ["Single layer", "Baseline vs predicted side-by-side"],
            horizontal=True,
            key=f"map_display_mode_{result_year}",
            help="Use side-by-side mode to compare the baseline salinity map and predicted salinity map with the same color scale.",
        )

        if display_mode == "Baseline vs predicted side-by-side":
            shared_range = salinity_range_from_columns(result_gdf, ["Salinity_Last_Year", "Predicted_Salinity"])
            st.caption(
                f"Both maps use the same salinity color range: "
                f"{shared_range[0]:.2f}–{shared_range[1]:.2f} g/L. "
                "Only the color scale is matched; prediction values are unchanged."
            )

            baseline_col, predicted_col = st.columns(2)
            baseline_event = None
            predicted_event = None

            with baseline_col:
                st.markdown("**Baseline salinity used by model**")
                baseline_map = build_interactive_map(
                    result_gdf,
                    "Salinity_Last_Year",
                    "Baseline salinity used by model (g/L)",
                    result_year,
                    fixed_vmin=shared_range[0],
                    fixed_vmax=shared_range[1],
                )
                baseline_event = st_folium(
                    baseline_map,
                    height=MAP_HEIGHT,
                    use_container_width=True,
                    returned_objects=["last_active_drawing"],
                    key=f"folium_compare_baseline_{result_year}",
                )

            with predicted_col:
                st.markdown(f"**Predicted salinity in {result_year}**")
                predicted_map = build_interactive_map(
                    result_gdf,
                    "Predicted_Salinity",
                    f"Predicted salinity in {result_year} (g/L)",
                    result_year,
                    fixed_vmin=shared_range[0],
                    fixed_vmax=shared_range[1],
                )
                predicted_event = st_folium(
                    predicted_map,
                    height=MAP_HEIGHT,
                    use_container_width=True,
                    returned_objects=["last_active_drawing"],
                    key=f"folium_compare_predicted_{result_year}",
                )

            selected_commune_card(predicted_event or baseline_event)

            with st.expander("Static PNG exports", expanded=False):
                export_col1, export_col2 = st.columns(2)
                with export_col1:
                    baseline_figure = draw_map(
                        result_gdf,
                        "Salinity_Last_Year",
                        "Baseline salinity used by model (g/L)",
                        result_year,
                        fixed_vmin=shared_range[0],
                        fixed_vmax=shared_range[1],
                    )
                    baseline_buffer = io.BytesIO()
                    baseline_figure.savefig(baseline_buffer, format="png", dpi=220, bbox_inches="tight")
                    baseline_figure.savefig(output_dir / "map_baseline_salinity_compare.png", dpi=220, bbox_inches="tight")
                    plt.close(baseline_figure)
                    st.download_button(
                        "Download baseline map PNG",
                        baseline_buffer.getvalue(),
                        file_name=f"salinity_{result_year}_baseline_compare.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_baseline_compare_{result_year}",
                    )
                with export_col2:
                    predicted_figure = draw_map(
                        result_gdf,
                        "Predicted_Salinity",
                        f"Predicted salinity in {result_year} (g/L)",
                        result_year,
                        fixed_vmin=shared_range[0],
                        fixed_vmax=shared_range[1],
                    )
                    predicted_buffer = io.BytesIO()
                    predicted_figure.savefig(predicted_buffer, format="png", dpi=220, bbox_inches="tight")
                    predicted_figure.savefig(output_dir / "map_predicted_salinity_compare.png", dpi=220, bbox_inches="tight")
                    plt.close(predicted_figure)
                    st.download_button(
                        "Download predicted map PNG",
                        predicted_buffer.getvalue(),
                        file_name=f"salinity_{result_year}_predicted_compare.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_predicted_compare_{result_year}",
                    )

            render_detailed_change_list(result_gdf, summary, result_year, expanded=True)

        else:
            map_col, list_col = st.columns([2.35, 1.0])

            with map_col:
                layer = st.radio(
                    "Map layer",
                    ["Absolute change (g/L)", "Percent change (%)", "Predicted salinity (g/L)", "Baseline salinity (g/L)"],
                    horizontal=True,
                    key=f"single_map_layer_{result_year}",
                )
                scale_mode = st.radio(
                    "Visualization value range",
                    ["Auto for selected layer", "Match baseline salinity range"],
                    horizontal=True,
                    key=f"single_map_scale_mode_{result_year}",
                    help=(
                        "Use 'Match baseline salinity range' when you want the predicted salinity "
                        "map and salinity plots to use the exact same value range as the baseline. "
                        "This changes only colors/axes, not prediction values."
                    ),
                )
                choices = {
                    "Absolute change (g/L)": ("absolute_change", f"Predicted {result_year} minus baseline salinity (g/L)"),
                    "Percent change (%)": ("percent_change", "Predicted change compared with baseline (%)"),
                    "Predicted salinity (g/L)": ("Predicted_Salinity", f"Predicted salinity in {result_year} (g/L)"),
                    "Baseline salinity (g/L)": ("Salinity_Last_Year", "Baseline salinity used by model (g/L)"),
                }
                field, title = choices[layer]
                fixed_range = None
                if scale_mode == "Match baseline salinity range" and field in {"Predicted_Salinity", "Salinity_Last_Year"}:
                    fixed_range = salinity_range_from_baseline(result_gdf)
                explain_scale_choice(scale_mode, field, fixed_range)
                interactive_map = build_interactive_map(
                    result_gdf,
                    field,
                    title,
                    result_year,
                    fixed_vmin=fixed_range[0] if fixed_range else None,
                    fixed_vmax=fixed_range[1] if fixed_range else None,
                )
                map_event = st_folium(
                    interactive_map,
                    height=MAP_HEIGHT,
                    use_container_width=True,
                    returned_objects=["last_active_drawing"],
                    key=f"folium_map_{result_year}_{field}",
                )
                selected_commune_card(map_event)

                with st.expander("Static PNG export", expanded=False):
                    figure = draw_map(
                        result_gdf,
                        field,
                        title,
                        result_year,
                        fixed_vmin=fixed_range[0] if fixed_range else None,
                        fixed_vmax=fixed_range[1] if fixed_range else None,
                    )
                    buffer = io.BytesIO()
                    figure.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
                    figure.savefig(output_dir / f"map_{field}.png", dpi=220, bbox_inches="tight")
                    plt.close(figure)
                    st.download_button(
                        "Download current map PNG",
                        buffer.getvalue(),
                        file_name=f"salinity_{result_year}_{field}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_single_map_{result_year}_{field}",
                    )

            with list_col:
                st.subheader("Detailed change list")
                render_detailed_change_list(result_gdf, summary, result_year, expanded=None)


    elif main_section == "Research / report plots":
        st.subheader("Research / report plots")
        st.caption("Plots are separated from the map so the map can stay large and the report figures have their own workspace.")
        match_baseline_plot_range = st.checkbox(
            "Use baseline salinity range for predicted salinity plots",
            value=False,
            help=(
                "When enabled, plots that compare baseline and predicted salinity use the "
                "baseline salinity min/max for the predicted axis or distribution range. "
                "This changes only visualization scale, not prediction values."
            ),
        )

        report_section = st.radio(
            "Report plot section",
            ["Overview", "Commune-level plots", "Model evaluation", "SHAP explainability"],
            horizontal=True,
            key="report_section",
        )

        if report_section == "Overview":
            p1, p2 = st.columns(2)
            with p1:
                sort_metric = st.selectbox("Province bar chart sort", ["percent_change", "absolute_change", "Province"], index=0)
                st.pyplot(draw_province_change_bar(summary, sort_metric), use_container_width=True)
            with p2:
                st.pyplot(draw_baseline_vs_predicted(summary, match_baseline_plot_range), use_container_width=True)

            p3, p4 = st.columns(2)
            with p3:
                st.pyplot(draw_commune_change_hist(result_gdf), use_container_width=True)
            with p4:
                st.pyplot(draw_salinity_distribution(result_gdf, match_baseline_plot_range), use_container_width=True)

            st.pyplot(draw_risk_category_bar(result_gdf), use_container_width=True)

        elif report_section == "Commune-level plots":
            st.markdown("**Commune-level spatial/statistical patterns**")
            top_n = st.slider("Top communes to show", min_value=10, max_value=30, value=15, step=5)
            cplot1, cplot2 = st.columns(2)
            with cplot1:
                st.pyplot(draw_top_commune_change_bar(result_gdf, n=top_n, largest=True), use_container_width=True)
            with cplot2:
                st.pyplot(draw_top_commune_change_bar(result_gdf, n=top_n, largest=False), use_container_width=True)

            cplot3, cplot4 = st.columns(2)
            with cplot3:
                st.pyplot(draw_commune_absolute_change_ranking(result_gdf, n=top_n), use_container_width=True)
            with cplot4:
                st.pyplot(draw_commune_baseline_vs_predicted(result_gdf, match_baseline_plot_range), use_container_width=True)

            st.pyplot(draw_boxplot_change_by_province(result_gdf), use_container_width=True)

        elif report_section == "Model evaluation":
            if "Actual_Salinity" not in result_gdf.columns or result_gdf["Actual_Salinity"].isna().all():
                st.info("No recorded Actual_Salinity values are available for evaluation plots.")
            else:
                eplot1, eplot2 = st.columns(2)
                with eplot1:
                    st.pyplot(draw_actual_vs_predicted_commune_scatter(result_gdf), use_container_width=True)
                with eplot2:
                    st.pyplot(draw_actual_vs_predicted_change_scatter(result_gdf), use_container_width=True)

                eplot3, eplot4 = st.columns(2)
                with eplot3:
                    st.pyplot(draw_error_distribution(result_gdf), use_container_width=True)
                with eplot4:
                    st.pyplot(draw_province_error_summary(result_gdf), use_container_width=True)

        elif report_section == "SHAP explainability":
            st.markdown("**SHAP explanation for the ONNX LSTM model**")
            st.caption(
                "This uses Kernel SHAP on a small commune sample, so it may take a little time. "
                "The graph explains which prepared CSV features push the salinity prediction up or down."
            )
            if model_frame is None:
                st.info("Run prediction again with this updated app before computing SHAP.")
            else:
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    shap_sample_size = st.number_input("Communes to explain", min_value=20, max_value=200, value=60, step=10)
                with s2:
                    shap_background_size = st.number_input("Background rows", min_value=10, max_value=80, value=25, step=5)
                with s3:
                    shap_nsamples = st.number_input("Kernel samples", min_value=50, max_value=500, value=120, step=25)
                with s4:
                    shap_top_n = st.number_input("Top features", min_value=5, max_value=30, value=15, step=5)

                shap_key = (
                    f"{signature}|shap|{int(shap_sample_size)}|{int(shap_background_size)}|"
                    f"{int(shap_nsamples)}|{_model_frame_signature(model_frame)}"
                )
                cached_shap = st.session_state.get("shap_result")
                shap_matches = isinstance(cached_shap, dict) and cached_shap.get("key") == shap_key

                if st.button("Compute / update SHAP graph", type="primary", use_container_width=True, on_click=keep_shap_section_open):
                    try:
                        with st.spinner("Computing Kernel SHAP explanation for the selected year…"):
                            frame_csv = model_frame.to_json(orient="split")
                            shap_payload = compute_kernel_shap_summary(
                                frame_csv=frame_csv,
                                model_path=saved_model_path,
                                pkl_path=saved_pkl_path,
                                sample_size=int(shap_sample_size),
                                background_size=int(shap_background_size),
                                nsamples=int(shap_nsamples),
                                random_seed=42,
                            )
                            st.session_state["shap_result"] = {"key": shap_key, "payload": shap_payload}
                            cached_shap = st.session_state["shap_result"]
                            shap_matches = True
                    except InputDataError as exc:
                        st.warning(str(exc))
                    except Exception as exc:
                        st.exception(exc)

                if not shap_matches:
                    st.info("Press **Compute / update SHAP graph** to generate SHAP plots for the current prediction result.")
                else:
                    shap_payload = cached_shap["payload"]
                    shap_summary = shap_payload["summary"]
                    shap_values = shap_payload["shap_values"]
                    sample_features = shap_payload["sample_features"]

                    st.pyplot(draw_shap_importance_bar(shap_summary, top_n=int(shap_top_n)), use_container_width=True)
                    st.pyplot(draw_shap_beeswarm_like(shap_values, sample_features, shap_summary, top_n=int(shap_top_n)), use_container_width=True)

                    top_feature_labels = shap_summary.head(int(shap_top_n))["feature_label"].tolist()
                    top_feature_names = shap_summary.head(int(shap_top_n))["feature"].tolist()
                    selected_label = st.selectbox("SHAP dependence feature", top_feature_labels)
                    selected_feature = top_feature_names[top_feature_labels.index(selected_label)]
                    st.pyplot(draw_shap_dependence(shap_values, sample_features, selected_feature), use_container_width=True)

                    st.download_button(
                        "Download SHAP feature importance CSV",
                        csv_bytes(shap_summary),
                        file_name=f"shap_feature_importance_{result_year}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    elif main_section == "Report tables / exports":
        st.subheader("Report tables and exports")
        st.markdown("**Province-level increase / decrease**")
        left, right = st.columns(2)
        number_config = {
            "percent_change": st.column_config.NumberColumn("Change (%)", format="%+.2f%%"),
            "absolute_change": st.column_config.NumberColumn("Change (g/L)", format="%+.3f"),
        }
        with left:
            st.caption("Largest province decreases")
            st.dataframe(
                summary.loc[summary["trend"] == "Decrease"].sort_values("percent_change"),
                hide_index=True,
                use_container_width=True,
                column_config=number_config,
                height=360,
            )
        with right:
            st.caption("Largest province increases")
            st.dataframe(
                summary.loc[summary["trend"] == "Increase"].sort_values("percent_change", ascending=False),
                hide_index=True,
                use_container_width=True,
                column_config=number_config,
                height=360,
            )

        st.markdown("**Commune-level report tables**")
        top_table_n = st.slider("Rows per report table", min_value=10, max_value=50, value=20, step=10)
        commune_report = make_compact_commune_list(result_gdf)
        t1, t2 = st.columns(2)
        with t1:
            st.caption("Top commune predicted increases")
            st.dataframe(
                sort_change_table(commune_report, "Largest increase (%)").head(top_table_n)[["Province", "District", "Commune", "Salinity", "Δ g/L", "Δ%", "Trend"]],
                hide_index=True,
                use_container_width=True,
                height=420,
            )
        with t2:
            st.caption("Top commune predicted decreases")
            st.dataframe(
                sort_change_table(commune_report, "Largest decrease (%)").head(top_table_n)[["Province", "District", "Commune", "Salinity", "Δ g/L", "Δ%", "Trend"]],
                hide_index=True,
                use_container_width=True,
                height=420,
            )

        st.download_button(
            "Download all commune predictions CSV",
            csv_bytes(pd.DataFrame(result_gdf.drop(columns="geometry"))),
            file_name=f"commune_predictions_{result_year}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Feature range warnings", expanded=False):
        if ranges.empty:
            st.success("All comparable inputs are within ranges observed in the prepared Final CSV.")
        else:
            st.warning("Some target-year inputs are outside the observed prepared CSV ranges.")
            st.dataframe(ranges, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    run_app()
