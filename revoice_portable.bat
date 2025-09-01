@echo off
setlocal enableextensions enabledelayedexpansion
chcp 65001 > nul
cd /d %~dp0

if not exist logs mkdir logs
set "LOGFILE=logs\revoice_portable.log"

if "%VERBOSE%"=="1" echo on

echo === Run started %DATE% %TIME%>>"%LOGFILE%"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv is required but not found.
    echo uv is required but not found.>>"%LOGFILE%"
    exit /b 1
)

if not exist venv\Scripts\python.exe (
    echo Creating local virtual environment...
    echo Creating local virtual environment...>>"%LOGFILE%"
    uv venv venv>>"%LOGFILE%" 2>&1 || (
        echo Failed to create virtual environment.
        echo Failed to create virtual environment.>>"%LOGFILE%"
        exit /b 1
    )
    echo Installing project requirements...
    echo Installing project requirements...>>"%LOGFILE%"
    uv pip install --python venv\Scripts\python.exe .>>"%LOGFILE%" 2>&1 || (
        echo Failed to install project requirements.
        echo Failed to install project requirements.>>"%LOGFILE%"
        exit /b 1
    )
    echo Installing torch with CUDA support...
    echo Installing torch with CUDA support...>>"%LOGFILE%"
    uv pip install --python venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu118>>"%LOGFILE%" 2>&1
    if errorlevel 1 (
        echo CUDA torch install failed, retrying CPU only...
        echo CUDA torch install failed, retrying CPU only...>>"%LOGFILE%"
        uv pip install --python venv\Scripts\python.exe torch>>"%LOGFILE%" 2>&1
    )
)

call venv\Scripts\activate.bat>>"%LOGFILE%" 2>&1 || (
    echo Failed to activate virtual environment.
    echo Failed to activate virtual environment.>>"%LOGFILE%"
    exit /b 1
)

for /f "delims=" %%i in ('python -c "import sys;print(sys.executable)"') do set "PYTHON_PATH=%%i"
echo Using Python: !PYTHON_PATH!>>"%LOGFILE%"
python -V>>"%LOGFILE%" 2>&1 || (
    echo Failed to query Python version.
    echo Failed to query Python version.>>"%LOGFILE%"
    exit /b 1
)

for /f "delims=" %%i in ('python -c "import json,importlib; m=importlib.util.find_spec(\"torch\"); ok=bool(m); torch=importlib.import_module(\"torch\") if ok else None; cu=importlib.import_module(\"torch.backends.cudnn\") if ok else None; print(json.dumps({\"ok\": ok, \"cuda\": torch.cuda.is_available() if ok else False, \"cudnn\": cu.is_available() if ok else False}))"') do set "TORCH_JSON=%%i"
echo !TORCH_JSON!>>"%LOGFILE%"

echo !TORCH_JSON! | find "\"ok\": false" >nul
if not errorlevel 1 (
    set "TTS_ENGINE=beep"
    echo Torch not available, falling back to beep TTS.
    echo Torch not available, falling back to beep TTS.>>"%LOGFILE%"
) else (
    echo !TORCH_JSON! | find "\"cuda\": true" >nul
    if not errorlevel 1 (
        echo Torch OK (CUDA)
        echo Torch OK (CUDA)>>"%LOGFILE%"
    ) else (
        echo Torch OK (CPU)
        echo Torch OK (CPU)>>"%LOGFILE%"
    )
)

if exist "%CD%\bin" set "PATH=%CD%\bin;%PATH%"
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin" set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin;%PATH%"
set "CT2_FORCE_CPU="

if not exist "%CD%\hf_cache" mkdir "%CD%\hf_cache"
set "HF_HOME=%CD%\hf_cache"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%"
set "TRANSFORMERS_CACHE=%HF_HOME%"

set "APP_CMD="
if exist ui\main.py (
    set "APP_CMD=python -m ui.main %*"
) else if exist src\ui\main.py (
    set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
    set "APP_CMD=python -m ui.main %*"
) else if exist main.py (
    set "APP_CMD=python main.py %*"
) else (
    for %%p in (revoice app ui cli) do (
        python -c "import %%p" >nul 2>&1
        if not errorlevel 1 (
            set "APP_CMD=python -m %%p %*"
            goto run_app
        )
    )
    echo Fatal: no entry point found.
    echo Fatal: no entry point found.>>"%LOGFILE%"
    echo Checked: ui\main.py, src\ui\main.py, main.py
    echo Checked: ui\main.py, src\ui\main.py, main.py>>"%LOGFILE%"
    echo Root listing:>>"%LOGFILE%"
    dir /b>>"%LOGFILE%" 2>&1
    if exist ui (
        echo UI listing:>>"%LOGFILE%"
        dir ui /b>>"%LOGFILE%" 2>&1
    )
    exit /b 1
)

:run_app
echo Starting: %APP_CMD%
echo Starting: %APP_CMD%>>"%LOGFILE%"
call %APP_CMD%>>"%LOGFILE%" 2>&1
set "APP_ERROR=%ERRORLEVEL%"

if not "%APP_ERROR%"=="0" (
    echo Application failed with code %APP_ERROR%.
    echo Application failed with code %APP_ERROR%.>>"%LOGFILE%"
    exit /b %APP_ERROR%
)

echo Application finished.
echo Application finished.>>"%LOGFILE%"
exit /b 0

