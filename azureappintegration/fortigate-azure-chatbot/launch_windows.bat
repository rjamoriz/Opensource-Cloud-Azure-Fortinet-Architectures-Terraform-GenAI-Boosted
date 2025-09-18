@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   🚀 FortiGate Azure Chatbot Launcher
echo   Windows ASUS ProArt GPU Optimized
echo ========================================

echo.
echo [1/4] Checking environment...

REM Check if virtual environment exists
if not exist "venv_gpu\Scripts\activate.bat" (
    echo ❌ GPU environment not found!
    echo Please run setup_windows_gpu.bat first
    pause
    exit /b 1
)

echo ✅ Virtual environment found

REM Activate virtual environment
echo [2/4] Activating GPU environment...
call venv_gpu\Scripts\activate.bat

REM Check GPU status
echo [3/4] Checking GPU status...
python -c "import torch; print(f'🎯 CUDA Available: {torch.cuda.is_available()}'); print(f'🔥 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Only\"}'); print(f'💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 0} GB')"

echo.
echo [4/4] Launching application...
echo 🌐 Starting Streamlit server on port 8080...
echo 📱 Open browser: http://localhost:8080
echo 🔬 Navigate to "Quantum Compression" for GPU acceleration
echo.

REM Launch with optimized settings
set STREAMLIT_SERVER_PORT=8080
set STREAMLIT_SERVER_ADDRESS=localhost
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

streamlit run src\app.py --server.port=8080 --server.address=localhost --browser.gatherUsageStats=false

echo.
echo Application closed. Press any key to exit...
pause >nul
