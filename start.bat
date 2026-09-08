@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment was not found.
    echo Run run_demo.bat first to create .venv and install packages.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate the project virtual environment.
    pause
    exit /b 1
)

python --version
python check_setup.py
if errorlevel 1 (
    echo.
    echo Setup check found missing inputs. The web app will show the exact missing files.
    echo.
)

python -m streamlit run app.py
pause
