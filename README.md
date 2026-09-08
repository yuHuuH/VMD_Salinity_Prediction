# Mekong Delta Salinity Prediction — CSV-Only ONNX Demo

This local Streamlit demo reads preprocessed rows directly from:

```text
data/VMD_Salinity_Dataset_IDW_500_Final.csv
```

It does **not** need a TIFF, raw weather CSV, or raw NOAA ONI file for years that already exist in the Final CSV.

## Required local files

```text
Mekong_Salinity_ONNX_CSV_Only_Demo/
│  app.py
│  salinity_pipeline.py
│  requirements.txt
│  run_demo.bat
│  start.bat
│
├─ trained_models/
│  ├─ best_lstm_model_top1.onnx
│  └─ best_lstm_package_top1.pkl
│
├─ boundary/
│  └─ gadm41_VNM_3.json
│
├─ data/
│  └─ VMD_Salinity_Dataset_IDW_500_Final.csv
│
└─ outputs/
```

The package already includes the Final CSV. Add only:

1. `best_lstm_model_top1.onnx`
2. `best_lstm_package_top1.pkl` containing `scaler_dyn` and `scaler_stat`
3. `gadm41_VNM_3.json`

## CSV feature logic

The Final CSV already contains these model features:

- `NDWI_mean`
- six monthly dynamic feature blocks from `M11_prev` through `M4`
- rainy-season feature summaries
- ONI fields
- `centroid_x`, `centroid_y`

The application recreates `Salinity_Last_Year` with the same training-notebook logic:

```python
history = history.sort_values(['Province', 'District', 'Commune', 'Year'])
history['Salinity_Last_Year'] = (
    history.groupby(['Province', 'District', 'Commune'])['Salinity_max'].shift(1)
)
```

`Salinity_max` for the selected year is not fed to ONNX. It is available only for the optional historical comparison panel.

## What a selectable year means

A year is selectable when its rows in the Final CSV contain all model feature columns and a prior salinity baseline from `groupby().shift(1)`.

This lets you replay inference for historical years stored in the Final CSV. It does **not** create a strict next-year forecast for a year that has no prepared CSV row. A true 2024 → 2025 forecast still requires retraining with lagged input/target years.

## Start on Windows

Install Python 3.11 (64-bit), then double-click:

```text
run_demo.bat
```

It creates `.venv/` inside the project folder. After setup, use `start.bat` for future runs.

## Main interface features

- Mainland-only Mekong Delta map; detached island groups are filtered before inference.
- Map on the left and detailed commune list on the right.
- Commune list contains all matching communes, with search, province filtering, and sorting.
- Compact rows: `Province | District | Commune | baseline → predicted | Δ g/L | Δ% | trend`.
- Province ranking, baseline-vs-predicted, and commune-change distribution charts.
- CSV and PNG downloads.

## Interactive map update

This package uses an interactive Folium map in Streamlit. Hover over a commune to highlight it and show province, district, commune, baseline salinity, predicted salinity, and salinity change. Click a commune to keep a popup open and show a selected-commune summary card below the map.

New dependencies: `folium` and `streamlit-folium` are included in `requirements.txt`.

## Build as Windows EXE

This package includes PyInstaller build scripts:

- `build_exe.bat` — recommended folder-based EXE.
- `build_exe_onefile.bat` — single EXE, slower startup.
- `launcher.py` — starts Streamlit from the EXE and opens the browser.

Run:

```bat
build_exe.bat
```

Then open:

```text
dist\MekongSalinityDemo\MekongSalinityDemo.exe
```

Keep the whole `dist\MekongSalinityDemo` folder together when moving it.
