@echo off
cd /d "%~dp0"

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo Python 3.11 was not found.
    echo Install Python 3.11 64-bit, then run this file again.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate the project virtual environment.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

python check_setup.py
if errorlevel 1 (
    echo.
    echo Setup check found missing inputs or no selectable CSV year.
    echo The web app will show the exact problem.
    echo.
)

python -m streamlit run app.py
pause
