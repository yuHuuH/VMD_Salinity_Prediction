@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Building Mekong Salinity Demo single EXE with SHAP support ^(slower startup^)
echo ============================================================

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Failed to create venv. Install Python 3.11 64-bit and try again.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -m ensurepip --upgrade
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m pip install -r requirements-build.txt

if exist build rmdir /s /q build
if exist dist\MekongSalinityDemo.exe del /q dist\MekongSalinityDemo.exe
if exist MekongSalinityDemo.spec del /q MekongSalinityDemo.spec

".venv\Scripts\python.exe" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --name MekongSalinityDemo ^
  --add-data "app.py;." ^
  --add-data "salinity_pipeline.py;." ^
  --add-data "data;data" ^
  --add-data "boundary;boundary" ^
  --add-data "trained_models;trained_models" ^
  --collect-all streamlit ^
  --collect-all streamlit_folium ^
  --collect-all folium ^
  --collect-all branca ^
  --collect-all geopandas ^
  --collect-all pyogrio ^
  --collect-all shapely ^
  --collect-all pyproj ^
  --collect-all onnxruntime ^
  --collect-all altair ^
  --collect-all pydeck ^
  --collect-all shap ^
  --collect-all numba ^
  --collect-all llvmlite ^
  --collect-all scipy ^
  --collect-all slicer ^
  --collect-all cloudpickle ^
  --collect-all tqdm ^
  --collect-all joblib ^
  --hidden-import shap.explainers._kernel ^
  --hidden-import shap.explainers._permutation ^
  --hidden-import shap.maskers._tabular ^
  --hidden-import sklearn.preprocessing._data ^
  --hidden-import sklearn.utils._cython_blas ^
  launcher.py

if errorlevel 1 (
    echo.
    echo Build failed.
    pause
    exit /b 1
)

echo.
echo Build complete.
echo Run this file:
echo dist\MekongSalinityDemo.exe
echo.
echo Onefile startup can be slow because it extracts Streamlit, GeoPandas, ONNX, CSV, and model files.
pause
