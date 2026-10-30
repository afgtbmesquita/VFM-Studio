@echo off
setlocal EnableExtensions
title abdXtract

rem Start abdXtract from this folder on any Windows PC.
rem Paths come from this file's location, not from a user name or drive letter.
rem Python is chosen in this order:
rem   1. VFMSTUDIO_PYTHON, when that interpreter already has the app's packages
rem   2. Generalized_VFM_GUI\.venv          (the README install)
rem   3. %LOCALAPPDATA%\vfmstudio\venv      (created on first run if needed)
rem First run needs Python 3.10+ (the "py" launcher or python on PATH)
rem and an internet connection. The virtual environment is not part of the git repo.

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "GUI=%ROOT%\Generalized_VFM_GUI"
set "CORE=%ROOT%\Generalized VFM"
set "USER_VENV=%LOCALAPPDATA%\vfmstudio\venv"
set "PY="
set "VENV="

set "GVFM_CORE_PATH=%CORE%"
set "PYTHONPATH=%GUI%\src"
set "PYTHONNOUSERSITE=1"
set "VFMSTUDIO_NO_VISTA="
set "QT_QPA_PLATFORM="

if not exist "%GUI%\pyproject.toml" (
  echo Could not find the VFM Studio application next to this script:
  echo   %GUI%
  goto :fail
)
if not exist "%CORE%\vfm_project.py" (
  echo Could not find the Generalized VFM solver next to this script:
  echo   %CORE%
  goto :fail
)

if defined VFMSTUDIO_PYTHON goto :try_override
goto :try_project_venv

:try_override
if not exist "%VFMSTUDIO_PYTHON%" (
  echo VFMSTUDIO_PYTHON does not exist:
  echo   %VFMSTUDIO_PYTHON%
  echo A local environment will be used instead.
  echo.
  goto :try_project_venv
)
call :python_ok "%VFMSTUDIO_PYTHON%"
if errorlevel 1 goto :override_unusable
call :deps_ok "%VFMSTUDIO_PYTHON%"
if errorlevel 1 goto :override_unusable
set "PY=%VFMSTUDIO_PYTHON%"
goto :launch

:override_unusable
echo VFMSTUDIO_PYTHON cannot run this app:
echo   %VFMSTUDIO_PYTHON%
echo A local environment will be used instead.
echo.
goto :try_project_venv

:try_project_venv
set "VENV=%GUI%\.venv"
set "PY=%VENV%\Scripts\python.exe"
if not exist "%PY%" goto :try_user_venv
call :python_ok "%PY%"
if errorlevel 1 (
  echo This project's virtual environment does not run on this computer.
  echo Removing it:
  echo   %VENV%
  echo.
  rmdir /s /q "%VENV%"
  goto :try_user_venv
)
goto :ensure_deps

:try_user_venv
set "VENV=%USER_VENV%"
set "PY=%VENV%\Scripts\python.exe"
if not exist "%PY%" goto :create_venv
call :python_ok "%PY%"
if errorlevel 1 (
  echo Removing a virtual environment that does not run on this computer:
  echo   %VENV%
  echo.
  rmdir /s /q "%VENV%"
  goto :create_venv
)
goto :ensure_deps

:create_venv
echo Setting up VFM Studio on this computer.
echo To use a Python install you already have, close this window, set
echo VFMSTUDIO_PYTHON to the full path of python.exe, and run this script again.
echo.
call :find_base
if errorlevel 1 goto :fail
set "VENV=%GUI%\.venv"
echo Creating a virtual environment:
echo   %VENV%
echo.
call :make_venv "%VENV%"
if not errorlevel 1 goto :created
echo.
echo Could not create that environment.
echo Creating one under your user profile instead:
echo   %USER_VENV%
echo.
set "VENV=%USER_VENV%"
call :make_venv "%VENV%"
if errorlevel 1 (
  echo.
  echo Could not create a Python virtual environment.
  goto :fail
)

:created
set "PY=%VENV%\Scripts\python.exe"
call :python_ok "%PY%"
if not errorlevel 1 goto :install_deps
echo The virtual environment was created but Python will not start:
echo   %PY%
if /i "%VENV%"=="%USER_VENV%" (
  echo.
  goto :fail
)
echo.
echo Creating one under your user profile instead:
echo   %USER_VENV%
echo.
rmdir /s /q "%VENV%"
set "VENV=%USER_VENV%"
call :make_venv "%VENV%"
if errorlevel 1 (
  echo.
  echo Could not create a Python virtual environment.
  goto :fail
)
goto :created

:ensure_deps
call :deps_ok "%PY%"
if errorlevel 1 goto :install_deps
call :stamp_ok
if errorlevel 1 goto :install_deps
goto :launch

:install_deps
echo.
echo Installing VFM Studio and its dependencies.
echo The first run needs an internet connection and can take several minutes.
echo.
"%PY%" -m pip install --upgrade pip
if errorlevel 1 echo Warning: pip could not be upgraded. Continuing.
echo.
"%PY%" -m pip install -e "%GUI%"
if errorlevel 1 (
  echo.
  echo Installation failed. See the messages above.
  goto :fail
)
call :deps_ok "%PY%"
if errorlevel 1 (
  echo.
  echo Packages installed, but Python still cannot import them.
  goto :fail
)
call :write_stamp

:launch
cd /d "%GUI%"
if errorlevel 1 (
  echo Could not open the application folder:
  echo   %GUI%
  goto :fail
)
echo Starting abdXtract...
echo Using Python:
echo   %PY%
echo Leave this window open while the app is running.
echo.
"%PY%" -m vfmstudio %*
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo abdXtract exited with code %EC%
  pause
)
exit /b %EC%

:fail
echo.
pause
exit /b 1

:python_ok
"%~1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

:deps_ok
"%~1" -c "import importlib.util,sys; mods=('PySide6','pyvista','numpy','scipy','h5py','pydantic','pyqtgraph','matplotlib','pandas'); sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

:make_venv
if defined USE_PY_LAUNCHER (
  py -3 -m venv "%~1"
) else (
  "%BASE_PY%" -m venv "%~1"
)
if errorlevel 1 exit /b 1
exit /b 0

:find_base
set "USE_PY_LAUNCHER="
set "BASE_PY="
where py >nul 2>&1
if not errorlevel 1 (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "USE_PY_LAUNCHER=1"
    exit /b 0
  )
)
for /f "delims=" %%P in ('where python 2^>nul ^& where python3 2^>nul') do (
  call :accept_base "%%P"
  if not errorlevel 1 (
    set "BASE_PY=%%P"
    goto :found_base
  )
)
echo Python 3.10 or newer was not found.
echo.
echo Install Python from https://www.python.org/downloads/
echo and turn on "Add python.exe to PATH", then run this script again.
exit /b 1

:found_base
exit /b 0

:accept_base
echo %~1 | findstr /i /c:"\WindowsApps\" >nul
if not errorlevel 1 exit /b 1
"%~1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

:stamp_ok
if not exist "%VENV%\vfmstudio-launch.stamp" exit /b 1
call :current_stamp
if errorlevel 1 exit /b 1
set "HAVE="
set /p HAVE=<"%VENV%\vfmstudio-launch.stamp"
if "%HAVE%"=="%STAMP%" exit /b 0
exit /b 1

:write_stamp
call :current_stamp
if errorlevel 1 exit /b 0
>"%VENV%\vfmstudio-launch.stamp" echo %STAMP%
exit /b 0

:current_stamp
call :file_hash "%GUI%\pyproject.toml"
if errorlevel 1 exit /b 1
set "PART_A=%HASH_OUT: =%"
call :file_hash "%GUI%\requirements.txt"
if errorlevel 1 exit /b 1
set "PART_B=%HASH_OUT: =%"
set "STAMP=%PART_A%%PART_B%"
exit /b 0

:file_hash
set "HASH_OUT="
for /f "usebackq delims=" %%H in (`certutil -hashfile "%~1" SHA256 ^| findstr /v ":"`) do (
  if not defined HASH_OUT set "HASH_OUT=%%H"
)
if not defined HASH_OUT exit /b 1
exit /b 0
