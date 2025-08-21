# Voice2Text Utility

This is a cross-platform command-line utility that allows you to record audio from your microphone, transcribe it using Whisper models (MLX on macOS, OpenAI Whisper on all platforms), and save the transcription to a file.

## Features

-   **Local Transcription:** Uses MLX framework on macOS for fast performance, or OpenAI Whisper for cross-platform support.
-   **Multiple Model Sizes:** Choose from tiny, base, small, medium, turbo, and large-v3-turbo models to balance speed and accuracy.
-   **Streaming Mode:** Continuous transcription for long-running sessions without disk space or memory issues.
-   **Real-time Audio Visualization:** Multiple visual styles during recording including Winamp-style frequency bars, waveforms, spectrum analyzer, and VU meters.
-   **Manual Recording Control:** Start and stop recording with a keypress.
-   **Immediate Recording:** Skip the "Press Enter to start" prompt with the `-i/--immediate` flag.
-   **Optional Editing:** Review and edit transcripts in your `$EDITOR` before saving.
-   **Flexible File Output:** Save to specific files or automatically timestamped files in chosen directories.

## Quick Start

**Prerequisites:** Internet connection (models download automatically, no Hugging Face account required)

To set up the project and install all dependencies:

### macOS with Apple Silicon (MLX support)
```bash
make setup
uv pip install -e ".[macos]"
```

### All Platforms (including Windows, Linux, Intel Macs)
```bash
make setup
uv pip install -e ".[all-platforms]"
```

To run the transcription utility:

```bash
make run
```

Alternatively, you can run the script directly using `uv` (after `make setup`):

```bash
uv run python voice2txt.py
```

Arguments can be passed directly to the script, for example:

```bash
uv run python voice2txt.py -e notes/my_audio.txt
```

## Model Selection

Choose the right model for your needs:

```bash
# Fast transcription (good for quick notes)
make run ARGS="-m tiny.en"

# Fast, lower accuracy option
make run ARGS="-m small.en"

# Default - best balance of speed and accuracy
make run

# Alternative high-quality option
make run ARGS="-m turbo"
```

### Setting Default Model via Environment Variable

You can customize the default model using the `WHISPER_DEFAULT_MODEL` environment variable:

```bash
# Set default to small.en for faster processing
export WHISPER_DEFAULT_MODEL=small.en
make run  # Now uses small.en by default

# Set default to tiny.en for quickest transcription
export WHISPER_DEFAULT_MODEL=tiny.en
make run  # Now uses tiny.en by default

# One-time override
WHISPER_DEFAULT_MODEL=turbo make run
```

**Valid options**: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `turbo`, `large-v3-turbo`

## Audio Visualization

The utility provides real-time audio visualization during recording to help you monitor input levels and frequency content. Choose from multiple visualization styles:

### Visualization Styles

**1. Frequency Bars (`--viz-style bars`)** - Classic Winamp-style spectrum analyzer
```
🔴 Recording... Press [Enter] to stop.
░░░░░█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░█░░█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░█░█░░█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
█░█░█░██░██░░█░░░░░░░░░░░░░░░░░░░░░░░░░░
```

**2. Waveform (`--viz-style waveform`)** - Amplitude visualization
```
🔴 Recording... Press [Enter] to stop.
███████████████░░█░██░░░▓▒░░░██▓▒▓█████████████░░░░░░
```

**3. Spectrum Analyzer (`--viz-style spectrum`)** - Horizontal frequency bands
```
🔴 Recording... Press [Enter] to stop.
██▓▒░░▒▓██░▒▓██▓▒░▓███▓▒░██▓▒░▒▓██░▒▓██▓▒░░░░░░░░
```

**4. VU Meter (`--viz-style vu`)** - Classic recording level meter (default)
```
🔴 Recording... Press [Enter] to stop.
[████████████▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░]
```

### Visualization Examples

```bash
# Winamp-style frequency bars (recommended for music/complex audio)
make run ARGS="--viz-style bars"

# Simple waveform (good for speech)
make run ARGS="--viz-style waveform"

# Horizontal spectrum analyzer
make run ARGS="--viz-style spectrum"

# Classic VU meter (default - no flag needed)
make run

# Disable visualization entirely
make run ARGS="--no-visualize"

# Combine with other options
make run ARGS="-i --viz-style bars -m tiny.en"
```

### Setting Default Output Folder

You can set a default folder for all transcriptions using the `WHISPER_OUTPUT_FOLDER` environment variable:

```bash
# Set default output folder
export WHISPER_OUTPUT_FOLDER=~/voice-notes
make run  # Saves to ~/voice-notes/transcription_2024-08-20_15-30-45.txt

# Create organized folders
export WHISPER_OUTPUT_FOLDER=~/Documents/Transcriptions
make run  # Auto-creates folder and saves with timestamp

# One-time override
WHISPER_OUTPUT_FOLDER=~/meeting-notes make run

# Still works with specific paths (overrides env var)
make run ARGS="important-call.txt"  # Saves to ./important-call.txt
make run ARGS="projects/"  # Saves to projects/ with timestamp
```

## Documentation

- **[Model Comparison Guide](MODEL_COMPARISON.md)** - Comprehensive comparison of all available models with performance charts, memory requirements, and usage recommendations
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines and project architecture details

## Usage Examples

```bash
# Record with default model and save with timestamp (to default folder if set)
make run

# Use default model and edit before saving
make run ARGS="-e"

# Start recording immediately (skip "Press Enter to start" prompt)
make run ARGS="-i"

# Save to specific file (overrides default folder)
make run ARGS="meeting-transcript.txt"

# Use faster model and save to specific directory
make run ARGS="-m small.en notes/"

# Combine immediate start with editing and specific folder
make run ARGS="-i -e notes/"

# Combine environment variables for personalized workflow
export WHISPER_DEFAULT_MODEL=small.en
export WHISPER_OUTPUT_FOLDER=~/quick-notes
make run  # Fast model, organized folder, no extra typing needed

# Quick capture workflow - immediate start with editing
make run ARGS="-i -e"

# Recording with visualization styles
make run ARGS="--viz-style bars"  # Winamp-style frequency bars
make run ARGS="--viz-style vu"    # Classic VU meter
make run ARGS="--no-visualize"    # No visualization

# Streaming mode for long sessions (memory-efficient)
make run ARGS="--stream"                    # Continuous transcription
make run ARGS="--stream -m tiny.en"        # Fast streaming with tiny model
make run ARGS="--stream --chunk-size 5"    # Faster processing (5-second chunks)
make run ARGS="--stream -i long-meeting.txt"  # All-day streaming session
```

## Streaming Mode

For long-running transcription sessions (meetings, lectures, all-day monitoring), use streaming mode to avoid memory and disk space issues:

```bash
# Basic streaming - processes 10-second chunks, maintains 30-second buffer
make run ARGS="--stream meeting-transcript.txt"

# Fast streaming with smaller model and quicker processing
make run ARGS="--stream -m tiny.en --chunk-size 5 --buffer-size 15"

# All-day streaming with immediate start
make run ARGS="--stream -i daily-log.txt"

# Streaming with custom visualization
make run ARGS="--stream --viz-style bars notes/"
```

**Streaming Benefits:**
- **Memory Efficient:** Uses circular buffer, never stores full recording
- **No Disk Limits:** Only saves transcription text, not audio files
- **Real-time Output:** See transcription appear as you speak with full date/time stamps
- **Configurable:** Adjust chunk size (1-30s) and buffer size (10-60s) for your needs
