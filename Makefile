PYTHON_VERSION=3.12.2
PYTHON_INTERPRETER=python

test_environment:
	@echo "Checking if Python interpreter is available..."
	@$(PYTHON_INTERPRETER) --version >nul 2>&1 || (echo "$(PYTHON_INTERPRETER) is not installed" && exit 1)
	@echo "$(PYTHON_INTERPRETER) is installed, checking version..."
	@$(PYTHON_INTERPRETER) -c "import sys; required_version = '$(PYTHON_VERSION)'; current_version = sys.version.split()[0]; assert current_version == required_version, f'Expected version {required_version}, but got {current_version}'" || (echo "Incorrect Python version" && exit 1)
	@echo "Python version is correct."

clean:
	for /r %%i in (*.pyc) do del /q "%%i"
	for /r %%i in (*.pyo) do del /q "%%i"
	for /d /r . %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d"
	rmdir /s /q "data/external" && mkdir "data/external" && echo. 2>data\external\.gitkeep
	rmdir /s /q "data/internal" && mkdir "data/internal" && echo. 2>data\internal\.gitkeep
	rmdir /s /q "data/processed" && mkdir "data/processed" && echo. 2>data\processed\.gitkeep
	rmdir /s /q "data/raw" && mkdir "data/raw" && echo. 2>data\raw\.gitkeep
	rmdir /s /q "logs" && mkdir "logs" && echo. 2>logs\.gitkeep
	rmdir /s /q "mlruns" && mkdir "mlruns" && echo. 2>mlruns\.gitkeep
	rmdir /s /q "models" && mkdir "models" && echo. 2>models\.gitkeep

lint:
	flake8 src