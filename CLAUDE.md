# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A cross-platform command-line utility for voice-to-text transcription. Uses MLX Whisper on macOS with Apple Silicon for optimal performance, or OpenAI Whisper on other platforms. Records audio from the microphone, transcribes it locally, and saves the text output.

## Development Commands

### Setup and Dependencies
- `make setup` - Initial setup: installs dependencies and auto-detects platform for appropriate Whisper library
- `uv run python voice2txt.py` - Run the script with uv environment management
- `make run ARGS="..."` - Run via Makefile (e.g., `make run ARGS="-e notes/"`)
- `make clean` - Remove virtual environment and generated files

### Common Usage Examples
- Record and save with timestamp: `make run`
- Edit before saving: `make run ARGS="-e"`
- Save to specific file: `make run ARGS="notes/meeting.txt"`
- Save to directory with timestamp: `make run ARGS="notes/"`
- Start recording immediately: `make run ARGS="-i"` (skips "Press Enter to start" prompt)
- Use different model: `make run ARGS="-m tiny"` (for speed) or `make run ARGS="-m small.en"` (faster) or default uses large-v3-turbo (best balance)
- Set custom default: `export WHISPER_DEFAULT_MODEL=small.en` then `make run` (uses small.en as default)
- Set default folder: `export WHISPER_OUTPUT_FOLDER=~/notes` then `make run` (saves to ~/notes with timestamp)
- Combine options: `make run ARGS="-i -e notes/"` (immediate start, edit before save, save to notes folder)

### Streaming Mode (Long-Running Sessions)
- Continuous transcription: `make run ARGS="--stream"` (memory-efficient, no disk storage)
- Fast streaming with small model: `make run ARGS="--stream -m tiny.en"`
- Custom chunk timing: `make run ARGS="--stream --chunk-size 5 --buffer-size 20"`
- All-day streaming: `make run ARGS="--stream -i meeting-log.txt"` (immediate start, saves timestamped chunks)

## Architecture

Single-file Python utility (`voice2txt.py`) with two modes:

### Traditional Mode (Default)
1. **Audio Recording**: Uses `sounddevice` to capture full audio to temporary file (16kHz, mono)
2. **Transcription**: Processes complete audio through Whisper model (MLX on macOS, OpenAI Whisper on other platforms)
3. **Optional Editing**: Opens transcription in `$EDITOR` if `-e` flag provided
4. **File Output**: Saves to specified path or timestamped file

### Streaming Mode (`--stream`)
1. **Continuous Audio Capture**: Circular buffer maintains 30-second audio window
2. **Chunked Processing**: Extracts 10-second chunks every 10 seconds for transcription
3. **Real-time Output**: Appends date/time stamped transcriptions directly to file as they're generated
4. **Memory Efficient**: Never stores full recording, only current buffer and chunks

Key implementation details:
- Threading for recording control and concurrent transcription processing
- Circular buffer (`collections.deque`) for memory-efficient audio storage
- Configurable chunk size (`--chunk-size`) and buffer size (`--buffer-size`)
- Optional memory monitoring with `psutil` (when available)
- Model selection with `-m` flag (tiny, base, small, medium, turbo, large-v3-turbo with .en variants)
- Environment variable support for default model (`WHISPER_DEFAULT_MODEL`)
- Environment variable support for default output folder (`WHISPER_OUTPUT_FOLDER`)

## Code Conventions

- **Python Style**: Type hints (`str | None`), snake_case naming
- **Error Handling**: Try/except with specific messages, `exit(1)` on failure
- **File Operations**: Context managers, explicit UTF-8 encoding
- **User Feedback**: Emoji prefixes (🔴 recording, ✅ success, ⚠️ warning)
- **Model Configuration**: Dynamic model selection with fallback to large-v3-turbo default

## Dependencies

- **Python Core**: sounddevice, soundfile, numpy
- **Whisper Libraries**: 
  - mlx-whisper (macOS with Apple Silicon)
  - openai-whisper (Windows, Linux, Intel Macs)
- **System**: uv (package management)
- **External**: Internet connection for automatic model downloads (Hugging Face account optional)
- remember to always update @README.md whenever new options are added or removed, or installation steps have changed.