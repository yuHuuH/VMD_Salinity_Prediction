"""CSV-only inference pipeline for the Mekong Delta ONNX salinity demo.

This version intentionally does NOT rebuild weather, ONI, or NDWI from raw files.
It uses feature rows already prepared in VMD_Salinity_Dataset_IDW_500_Final.csv,
matching the feature structure used by the user's LSTM training code.

Important:
- One CSV row = one commune + one year.
- The model receives all dynamic/static features from that selected-year CSV row.
- Salinity_Last_Year is recreated with the same groupby(...).shift(1) logic
  used in the supplied training notebook.
- The target year's recorded Salinity_max is never passed to ONNX. It can only
  be displayed as optional historical evaluation after inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import pickle
import re
import unicodedata
from typing import Any, Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

KEYS = ["Province", "District", "Commune"]
UTM_EPSG = 32648

MONTH_SPECS: list[tuple[str, int, int]] = [
    ("M11_prev", -1, 11),
    ("M12_prev", -1, 12),
    ("M1", 0, 1),
    ("M2", 0, 2),
    ("M3", 0, 3),
    ("M4", 0, 4),
]
DYNAMIC_BASES = [
    "rain", "t2m", "evap", "pres", "soil", "runoff",
    "wind_speed", "wind_dir", "wind", "ONI",
]
DYNAMIC_COLS = [[f"{base}_{tag}" for base in DYNAMIC_BASES] for tag, _, _ in MONTH_SPECS]
STATIC_COLS = [
    "centroid_x", "centroid_y", "NDWI_mean", "Salinity_Last_Year",
    "rain_Sum_MuaMua", "t2m_Mean_MuaMua", "evap_Sum_MuaMua",
    "runoff_Sum_MuaMua", "soil_Mean_MuaMua", "ONI_Rainy_LastYear",
]
FEATURES_REQUIRED = [*STATIC_COLS, *[column for group in DYNAMIC_COLS for column in group]]


class InputDataError(ValueError):
    """Clear user-facing input validation error."""


@dataclass
class Inventory:
    root: Path
    onnx_models: list[Path]
    pkl_packages: list[Path]
    boundaries: list[Path]
    history_csvs: list[Path]


# ---------------------------------------------------------------------
# File discovery / normalization
# ---------------------------------------------------------------------

def _is_ignored_path(path: Path) -> bool:
    ignored = {".venv", "__pycache__", ".git", "outputs"}
    return any(part in ignored for part in path.parts)


def project_files(root: Path) -> list[Path]:
    return [
        path for path in root.rglob("*")
        if path.is_file() and not _is_ignored_path(path.relative_to(root))
    ]


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", text)


def compact_name(value: object) -> str:
    return re.sub(r"[\s_\-]", "", normalize_name(value))


def make_key(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in KEYS if column not in frame.columns]
    if missing:
        raise InputDataError(f"Missing administrative columns: {', '.join(missing)}")
    out = frame.copy()
    out["_key"] = out[KEYS].apply(
        lambda row: "|".join(normalize_name(value) for value in row), axis=1
    )
    return out


def _read_header(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return list(pd.read_csv(path, nrows=1, encoding=encoding).columns)
        except Exception:
            continue
    return []


def _has_columns(columns: Iterable[str], required_aliases: Sequence[Sequence[str]]) -> bool:
    compact = {compact_name(column) for column in columns}
    return all(
        any(compact_name(alias) in compact for alias in aliases)
        for aliases in required_aliases
    )


def discover_inventory(root: str | Path) -> Inventory:
    root = Path(root).resolve()
    files = project_files(root)

    onnx_models = sorted(path for path in files if path.suffix.casefold() == ".onnx")
    pkl_packages = sorted(path for path in files if path.suffix.casefold() == ".pkl")

    boundaries = sorted(
        path for path in files
        if path.suffix.casefold() in {".json", ".geojson", ".gpkg"}
        and any(token in path.name.casefold() for token in ("gadm", "boundary", "ranh"))
    )
    if not boundaries:
        boundaries = sorted(
            path for path in files
            if path.suffix.casefold() in {".json", ".geojson", ".gpkg"}
        )

    history_csvs: list[Path] = []
    for path in files:
        if path.suffix.casefold() != ".csv":
            continue
        if _has_columns(
            _read_header(path),
            [["year"], ["province"], ["district"], ["commune"], ["salinity_max", "salinitymax"]],
        ):
            history_csvs.append(path)

    return Inventory(
        root=root,
        onnx_models=onnx_models,
        pkl_packages=pkl_packages,
        boundaries=boundaries,
        history_csvs=sorted(history_csvs),
    )


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    last_exc: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_exc = exc
    raise InputDataError(f"Could not read CSV '{path.name}': {last_exc}")


def _find_column(columns: Iterable[str], aliases: Sequence[str]) -> str | None:
    lookup = {compact_name(column): str(column) for column in columns}
    for alias in aliases:
        found = lookup.get(compact_name(alias))
        if found:
            return found
    return None


def _rename_aliases(frame: pd.DataFrame, mapping: dict[str, Sequence[str]]) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for canonical, aliases in mapping.items():
        found = _find_column(frame.columns, aliases)
        if found and found != canonical:
            rename[found] = canonical
    return frame.rename(columns=rename).copy()


def canonical_history(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "Year": ["Year", "year"],
        "Province": ["Province", "NAME_1", "province_name"],
        "District": ["District", "NAME_2", "district_name"],
        "Commune": ["Commune", "NAME_3", "commune_name", "ward"],
        "Salinity_max": ["Salinity_max", "salinity max", "salinitymax"],
    }
    out = _rename_aliases(frame, aliases)

    required_base = ["Year", *KEYS, "Salinity_max"]
    missing_base = [column for column in required_base if column not in out.columns]
    if missing_base:
        raise InputDataError(f"Final CSV is missing: {', '.join(missing_base)}")

    # Every feature except Salinity_Last_Year must already exist in the Final CSV.
    required_csv_features = [column for column in FEATURES_REQUIRED if column != "Salinity_Last_Year"]
    missing_features = [column for column in required_csv_features if column not in out.columns]
    if missing_features:
        raise InputDataError(
            "This CSV cannot be used for ONNX inference because it is missing prepared model feature columns: "
            + ", ".join(missing_features)
        )

    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["Salinity_max"] = pd.to_numeric(out["Salinity_max"], errors="coerce")
    out = out.dropna(subset=["Year", *KEYS, "Salinity_max"]).copy()
    out["Year"] = out["Year"].astype(int)

    numeric_columns = ["Salinity_max", *required_csv_features]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    return make_key(out)


# ---------------------------------------------------------------------
# Boundary / mainland-only policy
# ---------------------------------------------------------------------

def read_boundary(path: str | Path) -> gpd.GeoDataFrame:
    try:
        communes = gpd.read_file(path)
    except Exception as exc:
        raise InputDataError(f"Could not read boundary file '{Path(path).name}': {exc}") from exc
    if communes.crs is None:
        raise InputDataError("The boundary file has no CRS. Use a GADM GeoJSON/GPKG with CRS metadata.")

    aliases = {
        "Province": ["Province", "NAME_1", "name_1", "province_name"],
        "District": ["District", "NAME_2", "name_2", "district_name"],
        "Commune": ["Commune", "NAME_3", "name_3", "commune_name", "ward"],
    }
    communes = _rename_aliases(communes, aliases)
    missing = [column for column in KEYS if column not in communes.columns]
    if missing:
        raise InputDataError(f"Boundary file is missing: {', '.join(missing)}")

    communes = communes.dropna(subset=KEYS).copy()
    communes = communes[communes.geometry.notna() & ~communes.geometry.is_empty].copy()
    communes = communes.dissolve(by=KEYS, as_index=False)
    return make_key(communes)


def keep_largest_polygon_part(geometry):
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.geom_type == "Polygon":
        return geometry
    if geometry.geom_type == "MultiPolygon":
        return max(geometry.geoms, key=lambda part: part.area)
    if geometry.geom_type == "GeometryCollection":
        parts = []
        for part in geometry.geoms:
            if part.geom_type == "Polygon":
                parts.append(part)
            elif part.geom_type == "MultiPolygon":
                parts.extend(list(part.geoms))
        return max(parts, key=lambda part: part.area) if parts else geometry
    return geometry


def keep_mainland_mekong_component(communes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove detached islands by retaining the largest connected mainland component."""
    if communes.empty:
        return communes.copy()

    work = communes.copy()
    work["geometry"] = work.geometry.apply(keep_largest_polygon_part)
    work = work[work.geometry.notna() & ~work.geometry.is_empty].copy().reset_index(drop=True)
    if len(work) <= 1:
        return work

    try:
        neighbours = gpd.sjoin(
            work[["geometry"]],
            work[["geometry"]],
            how="inner",
            predicate="intersects",
            lsuffix="left",
            rsuffix="right",
        )
    except Exception as exc:
        raise InputDataError(
            "Could not apply the mainland-only boundary filter. "
            f"Details: {exc}"
        ) from exc

    parent = list(range(len(work)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, right_index in zip(neighbours.index.to_numpy(), neighbours["index_right"].to_numpy()):
        union(int(left_index), int(right_index))

    work["_mainland_component"] = [find(index) for index in range(len(work))]
    projected = work.to_crs(epsg=UTM_EPSG)
    areas = (
        projected.assign(_area_m2=projected.geometry.area)
        .groupby("_mainland_component")["_area_m2"]
        .sum()
    )
    largest_component = areas.idxmax()
    return work.loc[work["_mainland_component"] == largest_component].drop(columns="_mainland_component").copy()


# ---------------------------------------------------------------------
# CSV-only feature preparation
# ---------------------------------------------------------------------

def add_salinity_last_year(history: pd.DataFrame) -> pd.DataFrame:
    """Mirror the exact training-notebook groupby().shift(1) feature logic."""
    ordered = history.sort_values([*KEYS, "Year"], kind="stable").copy()
    grouped = ordered.groupby(KEYS, sort=False)
    ordered["Salinity_Last_Year"] = grouped["Salinity_max"].shift(1)
    ordered["Baseline_Year"] = grouped["Year"].shift(1)
    return ordered


def valid_year_table(history: pd.DataFrame) -> pd.DataFrame:
    prepared = add_salinity_last_year(history)
    rows: list[dict[str, Any]] = []

    for year in sorted(prepared["Year"].unique().tolist()):
        target = prepared.loc[prepared["Year"] == int(year)].copy()
        eligible = target.dropna(subset=FEATURES_REQUIRED)
        rows.append({
            "target_year": int(year),
            "status": "Valid" if not eligible.empty else "Incomplete",
            "csv_rows": int(len(target)),
            "eligible_rows": int(len(eligible)),
            "reason": (
                "Prepared feature rows and prior-salinity baseline are available in Final CSV."
                if not eligible.empty
                else "No rows contain all required CSV features plus Salinity_Last_Year."
            ),
        })

    return pd.DataFrame(rows, columns=["target_year", "status", "csv_rows", "eligible_rows", "reason"])


def build_model_frame_from_csv(
    history: pd.DataFrame,
    communes: gpd.GeoDataFrame,
    target_year: int,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, list[int]]:
    """Create ONNX input rows solely from prepared Final CSV features."""
    prepared = add_salinity_last_year(history)
    target = prepared.loc[prepared["Year"] == int(target_year)].copy()
    if target.empty:
        raise InputDataError(f"No CSV rows exist for target year {target_year}.")

    # Remove rows that cannot be sent to the model. Salinity_max remains in the
    # DataFrame only as optional recorded actual value; it is never an ONNX input.
    target = target.dropna(subset=FEATURES_REQUIRED).copy()
    if target.empty:
        raise InputDataError(
            f"Target year {target_year} has no rows with complete model features and a prior salinity baseline."
        )

    target_keys = set(target["_key"].unique())
    display_communes = communes.loc[communes["_key"].isin(target_keys)].copy()
    if display_communes.empty:
        raise InputDataError("No commune names in the boundary match the selected target-year CSV rows.")

    display_communes = keep_mainland_mekong_component(display_communes)
    mainland_keys = set(display_communes["_key"].unique())
    target = target.loc[target["_key"].isin(mainland_keys)].copy()

    if target.empty:
        raise InputDataError("No mainland commune rows remained after matching the boundary.")

    # Preserve required model feature order and make numeric conversion explicit.
    for column in FEATURES_REQUIRED:
        target[column] = pd.to_numeric(target[column], errors="coerce")
    missing = target[FEATURES_REQUIRED].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise InputDataError(
            "Selected CSV rows have missing model feature values: "
            + ", ".join(f"{column} ({count})" for column, count in missing.items())
        )

    target["Actual_Salinity"] = pd.to_numeric(target["Salinity_max"], errors="coerce")
    target["Baseline_Year"] = pd.to_numeric(target["Baseline_Year"], errors="coerce").astype(int)
    baseline_years = sorted(target["Baseline_Year"].unique().tolist())

    keep_columns = [
        "_key", *KEYS, "Year", "Baseline_Year", "Actual_Salinity",
        *FEATURES_REQUIRED,
    ]
    model_frame = target[keep_columns].copy()
    display_communes = display_communes.loc[display_communes["_key"].isin(model_frame["_key"])].copy()

    return model_frame, display_communes, baseline_years


# ---------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------

def make_lstm_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_dynamic = np.stack(
        [frame[group].to_numpy(dtype=np.float32) for group in DYNAMIC_COLS],
        axis=1,
    )
    x_static = frame[STATIC_COLS].to_numpy(dtype=np.float32)
    return x_dynamic, x_static


def _shape_dim_equals(value: Any, expected: int) -> bool:
    try:
        return int(value) == expected
    except (TypeError, ValueError):
        return False


def _resolve_onnx_inputs(session: Any) -> tuple[str, str]:
    nodes = list(session.get_inputs())
    dynamic_candidates = [
        node for node in nodes
        if len(node.shape) == 3
        and _shape_dim_equals(node.shape[-2], 6)
        and _shape_dim_equals(node.shape[-1], 10)
    ]
    static_candidates = [
        node for node in nodes
        if len(node.shape) == 2 and _shape_dim_equals(node.shape[-1], 10)
    ]

    if len(dynamic_candidates) != 1:
        dynamic_candidates = [node for node in nodes if len(node.shape) == 3]
    if len(static_candidates) != 1:
        static_candidates = [node for node in nodes if len(node.shape) == 2]

    if len(dynamic_candidates) != 1 or len(static_candidates) != 1:
        found = "; ".join(f"{node.name}: {node.shape}" for node in nodes)
        raise InputDataError(
            "The ONNX model must expose one dynamic input (batch, 6, 10) and "
            f"one static input (batch, 10). Found: {found}"
        )
    return dynamic_candidates[0].name, static_candidates[0].name


def load_lstm_artifacts(model_path: str | Path, package_path: str | Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise InputDataError("onnxruntime is not installed. Run run_demo.bat to install project dependencies.") from exc

    model_path, package_path = Path(model_path), Path(package_path)
    if model_path.suffix.casefold() != ".onnx":
        raise InputDataError(f"Expected .onnx model, received '{model_path.name}'.")

    try:
        with open(package_path, "rb") as handle:
            package = pickle.load(handle)
    except Exception as exc:
        raise InputDataError(f"Could not read scaler package '{package_path.name}': {exc}") from exc
    if not isinstance(package, dict) or "scaler_dyn" not in package or "scaler_stat" not in package:
        raise InputDataError("The .pkl package must contain scaler_dyn and scaler_stat.")

    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        dynamic_name, static_name = _resolve_onnx_inputs(session)
    except InputDataError:
        raise
    except Exception as exc:
        raise InputDataError(f"Could not load ONNX model '{model_path.name}': {exc}") from exc

    return session, package, dynamic_name, static_name


def predict_lstm(frame: pd.DataFrame, model_path: str | Path, package_path: str | Path) -> np.ndarray:
    session, package, dynamic_name, static_name = load_lstm_artifacts(model_path, package_path)
    x_dynamic, x_static = make_lstm_arrays(frame)

    try:
        x_dynamic_scaled = package["scaler_dyn"].transform(
            x_dynamic.reshape(-1, x_dynamic.shape[-1])
        ).reshape(x_dynamic.shape).astype(np.float32, copy=False)
        x_static_scaled = package["scaler_stat"].transform(x_static).astype(np.float32, copy=False)
    except Exception as exc:
        raise InputDataError(
            "Saved scalers could not transform the prepared CSV features. Ensure .onnx and .pkl are from the same training run. "
            f"Details: {exc}"
        ) from exc

    try:
        outputs = session.run(None, {dynamic_name: x_dynamic_scaled, static_name: x_static_scaled})
        prediction = np.asarray(outputs[0]).reshape(-1)
    except Exception as exc:
        raise InputDataError(f"ONNX inference failed: {exc}") from exc

    if len(prediction) != len(frame):
        raise InputDataError(f"ONNX output rows ({len(prediction)}) do not match prepared CSV rows ({len(frame)}).")
    return prediction.astype(float)


# ---------------------------------------------------------------------
# Results / visualization support
# ---------------------------------------------------------------------

def make_results_gdf(
    communes: gpd.GeoDataFrame,
    frame: pd.DataFrame,
    predictions: np.ndarray,
    target_year: int,
) -> gpd.GeoDataFrame:
    result = frame[["_key", "Baseline_Year", "Salinity_Last_Year", "Actual_Salinity"]].copy()
    result["Target_Year"] = int(target_year)
    result["Predicted_Salinity"] = predictions
    result["absolute_change"] = result["Predicted_Salinity"] - result["Salinity_Last_Year"]
    result["percent_change"] = np.where(
        np.abs(result["Salinity_Last_Year"]) > 1e-6,
        result["absolute_change"] / result["Salinity_Last_Year"] * 100.0,
        np.nan,
    )
    result["trend"] = np.select(
        [result["absolute_change"] > 0.01, result["absolute_change"] < -0.01],
        ["Increase", "Decrease"],
        default="Stable",
    )

    display = communes.loc[communes["_key"].isin(result["_key"])].copy()
    display["geometry"] = display.geometry.apply(keep_largest_polygon_part)
    display = display[display.geometry.notna() & ~display.geometry.is_empty].copy()
    return display.merge(result, on="_key", how="inner")


def province_summary(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby("Province", as_index=False).agg(
        communes=("Commune", "size"),
        baseline_salinity=("Salinity_Last_Year", "mean"),
        predicted_salinity=("Predicted_Salinity", "mean"),
        increase_communes=("trend", lambda values: int((values == "Increase").sum())),
        decrease_communes=("trend", lambda values: int((values == "Decrease").sum())),
    )
    grouped["absolute_change"] = grouped["predicted_salinity"] - grouped["baseline_salinity"]
    grouped["percent_change"] = np.where(
        np.abs(grouped["baseline_salinity"]) > 1e-6,
        grouped["absolute_change"] / grouped["baseline_salinity"] * 100.0,
        np.nan,
    )
    grouped["trend"] = np.select(
        [grouped["absolute_change"] > 0.01, grouped["absolute_change"] < -0.01],
        ["Increase", "Decrease"],
        default="Stable",
    )
    return grouped.sort_values("percent_change", ascending=False, kind="stable").reset_index(drop=True)


def evaluate_against_recorded(results: gpd.GeoDataFrame) -> dict[str, float]:
    valid = results.dropna(subset=["Actual_Salinity", "Predicted_Salinity"])
    if valid.empty:
        raise InputDataError("The selected CSV rows contain no recorded Salinity_max values for evaluation.")
    error = valid["Predicted_Salinity"] - valid["Actual_Salinity"]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denominator = float(np.sum((valid["Actual_Salinity"] - valid["Actual_Salinity"].mean()) ** 2))
    r2 = float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "n": int(len(valid))}


def training_range_report(history: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    prepared_history = add_salinity_last_year(history)
    rows: list[dict[str, Any]] = []
    for column in FEATURES_REQUIRED:
        training_values = pd.to_numeric(prepared_history[column], errors="coerce").dropna()
        current_values = pd.to_numeric(frame[column], errors="coerce")
        if training_values.empty:
            continue
        minimum, maximum = float(training_values.min()), float(training_values.max())
        out_count = int(((current_values < minimum) | (current_values > maximum)).sum())
        if out_count:
            rows.append({
                "feature": column,
                "training_min": minimum,
                "training_max": maximum,
                "out_of_range_communes": out_count,
            })
    return pd.DataFrame(rows).sort_values("out_of_range_communes", ascending=False) if rows else pd.DataFrame(
        columns=["feature", "training_min", "training_max", "out_of_range_communes"]
    )


def save_outputs(root: str | Path, results: gpd.GeoDataFrame, summary: pd.DataFrame, metadata: dict[str, Any]) -> Path:
    root = Path(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_year = metadata.get("target_year", "unknown")
    out_dir = root / "outputs" / f"prediction_{target_year}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.drop(columns="geometry", errors="ignore").to_csv(
        out_dir / "commune_predictions.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(out_dir / "province_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return out_dir
