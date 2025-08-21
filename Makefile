# Makefile for the voice2txt project

# --- Variables ---
SHELL := /bin/bash
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python

# --- Targets ---

.PHONY: help setup check-uv run uv-run clean

help:
	@echo "Usage:"
	@echo "  make setup          - Sets up the project with Python dependencies (requires uv)."
	@echo "  make run ARGS=\"-e\"  - Runs the script. Pass arguments with ARGS, e.g., ARGS=\"\"-e notes/\"\"."
	@echo "  make uv-run ARGS=\"-e\" - Runs the script using 'uv run'. Pass arguments with ARGS."
	@echo "  make clean          - Removes generated files and directories."
	@echo "  make check-uv       - Checks if uv is installed and provides installation instructions."

setup: check-uv $(VENV_DIR)
	@echo "✨ Setup complete. You can now run the script with 'make run' or 'make uv-run'"

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "⚠️  uv is not installed. Please install it using one of these methods:"; \
		echo ""; \
		echo "  macOS/Linux:"; \
		echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo ""; \
		echo "  macOS (with Homebrew):"; \
		echo "    brew install uv"; \
		echo ""; \
		echo "  Windows (PowerShell):"; \
		echo "    powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""; \
		echo ""; \
		echo "  Or with pip/pipx:"; \
		echo "    pip install uv  # or pipx install uv"; \
		echo ""; \
		echo "  For more options, visit: https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	}

# Create Python virtual environment and install Python dependencies
$(VENV_DIR):
	@echo "🐍 Creating Python virtual environment with uv..."
	@uv venv $(VENV_DIR)
	@echo "🐍 Installing Python dependencies..."
	@uv pip install sounddevice soundfile numpy --python $(PYTHON)
	@echo "🐍 Detecting platform and installing Whisper library..."
	@if [ "$$(uname)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ]; then \
		echo "🍎 macOS with Apple Silicon detected. Installing MLX Whisper..."; \
		uv pip install mlx-whisper --python $(PYTHON); \
	else \
		echo "💻 Installing OpenAI Whisper for cross-platform support..."; \
		uv pip install openai-whisper --python $(PYTHON); \
	fi
	@echo "\n\n**********************************************************************"
	@echo "IMPORTANT: You may need to log in to Hugging Face to download the model."
	@echo "If the script fails, run this command and enter your HF token:"
	@echo "$(PYTHON) -m huggingface_hub login"
	@echo "**********************************************************************\n"

run:
	@$(PYTHON) voice2txt.py $(ARGS)

uv-run:
	@uv run python voice2txt.py $(ARGS)

clean:
	@echo "🔥 Removing generated files and directories..."
	@rm -rf $(VENV_DIR) __pycache__
	@rm -f transcription_*.txt *.pyc
