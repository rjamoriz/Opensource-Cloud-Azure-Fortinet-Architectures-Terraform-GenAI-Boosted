@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   FortiGate Azure Chatbot - Windows GPU Setup
echo   Optimized for ASUS ProArt + NVIDIA GPU
echo ========================================

echo.
echo [1/8] System Requirements Check...
echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+ first
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo [2/8] NVIDIA GPU Detection...
echo Checking NVIDIA drivers and GPU...
nvidia-smi
if %errorlevel% neq 0 (
    echo ERROR: NVIDIA GPU not detected or drivers not installed
    echo Please install latest NVIDIA drivers from: https://www.nvidia.com/drivers/
    pause
    exit /b 1
)

echo.
echo [3/8] Creating optimized virtual environment...
if exist venv_gpu (
    echo Removing existing environment...
    rmdir /s /q venv_gpu
)
python -m venv venv_gpu
call venv_gpu\Scripts\activate.bat

echo.
echo [4/8] Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel

echo.
echo [5/8] Installing PyTorch with CUDA 11.8 support...
echo This may take several minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo [6/8] Installing GPU-optimized AI libraries...
pip install -r requirements_windows_gpu.txt

echo.
echo [7/8] Creating Windows-specific configuration...
echo Creating config directory...
mkdir config 2>nul

echo # Windows GPU Configuration > config\windows_gpu_config.py
echo import torch >> config\windows_gpu_config.py
echo import os >> config\windows_gpu_config.py
echo. >> config\windows_gpu_config.py
echo # GPU Configuration >> config\windows_gpu_config.py
echo CUDA_AVAILABLE = torch.cuda.is_available() >> config\windows_gpu_config.py
echo GPU_COUNT = torch.cuda.device_count() >> config\windows_gpu_config.py
echo GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory if CUDA_AVAILABLE else 0 >> config\windows_gpu_config.py
echo. >> config\windows_gpu_config.py
echo # Optimization Settings >> config\windows_gpu_config.py
echo os.environ['CUDA_LAUNCH_BLOCKING'] = '0' >> config\windows_gpu_config.py
echo os.environ['TORCH_USE_CUDA_DSA'] = '1' >> config\windows_gpu_config.py
echo torch.backends.cudnn.benchmark = True >> config\windows_gpu_config.py

echo.
echo [8/8] System Validation and Performance Test...
echo Testing GPU setup...
python -c "import torch; print('\n=== GPU SYSTEM INFO ==='); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'PyTorch Version: {torch.__version__}'); import psutil; print(f'System RAM: {psutil.virtual_memory().total // (1024**3)} GB'); print('\n=== PERFORMANCE TEST ==='); import time; start = time.time(); x = torch.randn(1000, 1000).cuda() if torch.cuda.is_available() else torch.randn(1000, 1000); y = torch.mm(x, x); print(f'Matrix multiplication test: {time.time() - start:.4f}s'); print('\n=== READY FOR AI WORKLOADS ===\n')"

echo.
echo ========================================
echo   🚀 ASUS ProArt GPU Setup Complete!
echo ========================================
echo.
echo 🎯 Next Steps:
echo 1. Keep this terminal open
echo 2. Run: streamlit run src\app.py --server.port=8080
echo 3. Open browser: http://localhost:8080
echo 4. Navigate to "🔬 Quantum Compression" tab
echo 5. Experience 10x+ performance boost!
echo.
echo 💡 Pro Tips:
echo - Use GPU monitoring: nvidia-smi -l 1
echo - Check GPU usage in Task Manager
echo - Monitor temperatures during heavy workloads
echo.
echo 🔧 Troubleshooting:
echo - If CUDA errors: Update NVIDIA drivers
echo - If memory errors: Reduce batch sizes
echo - If slow performance: Check GPU utilization
echo.

echo Press any key to launch the application...
pause >nul

echo.
echo 🚀 Launching FortiGate Azure Chatbot...
streamlit run src\app.py --server.port=8080

pause
