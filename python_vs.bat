@echo off
rem Запуск Python в окружении Visual Studio Build Tools (нужно для JIT-компиляции
rem ядер MIOpen/hipRTC на Windows-ROCm: без MSVC-заголовков компиляция падает).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
"%~dp0.venv\Scripts\python.exe" %*
