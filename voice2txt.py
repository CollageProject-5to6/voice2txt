import argparse
import datetime
import os
import platform
import subprocess
import sys
import tempfile
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading


# --- Platform Detection and Library Import ---
def get_whisper_backend():
    """Detect platform and return appropriate Whisper backend."""
    system = platform.system().lower()

    if system == "darwin":  # macOS
        try:
            from mlx_whisper.transcribe import transcribe as mlx_transcribe

            return "mlx", mlx_transcribe
        except ImportError:
            pass

    # Fallback to OpenAI Whisper for all platforms (including macOS if MLX fails)
    try:
        import whisper

        return "openai-whisper", whisper
    except ImportError:
        raise ImportError(
            "No compatible Whisper library found. Please install either:\n"
            "  - mlx-whisper (macOS with Apple Silicon)\n"
            "  - openai-whisper (all platforms)"
        )


WHISPER_BACKEND, whisper_lib = get_whisper_backend()

# --- Configuration ---
# Default model configuration based on backend
DEFAULT_MODELS = {
    "mlx": "mlx-community/whisper-large-v3-turbo",
    "openai-whisper": "large-v3-turbo",
}

# Model size mappings for both backends
MODEL_MAPPINGS = {
    "tiny": {"mlx": "mlx-community/whisper-tiny", "openai-whisper": "tiny"},
    "tiny.en": {
        "mlx": "mlx-community/whisper-tiny.en-mlx-q4",
        "openai-whisper": "tiny.en",
    },
    "base": {"mlx": "mlx-community/whisper-base-mlx", "openai-whisper": "base"},
    "base.en": {
        "mlx": "mlx-community/whisper-base.en-mlx",
        "openai-whisper": "base.en",
    },
    "small": {"mlx": "mlx-community/whisper-small-mlx", "openai-whisper": "small"},
    "small.en": {
        "mlx": "mlx-community/whisper-small.en-mlx",
        "openai-whisper": "small.en",
    },
    "medium": {
        "mlx": "mlx-community/whisper-medium-mlx-fp32",
        "openai-whisper": "medium",
    },
    "medium.en": {
        "mlx": "mlx-community/whisper-medium.en-mlx-4bit",
        "openai-whisper": "medium.en",
    },
    "turbo": {"mlx": "mlx-community/whisper-turbo", "openai-whisper": "turbo"},
    "large-v3-turbo": {
        "mlx": "mlx-community/whisper-large-v3-turbo",
        "openai-whisper": "large-v3-turbo",
    },
}


def get_default_model() -> str:
    """Get the default model from environment variable or fallback to built-in default."""
    env_model = os.environ.get("WHISPER_DEFAULT_MODEL")
    if env_model and env_model in MODEL_MAPPINGS:
        return env_model
    return "large-v3-turbo"  # Built-in default


def get_model_name(model_size: str) -> str:
    """Get the appropriate model name for the current backend."""
    if model_size in MODEL_MAPPINGS:
        return MODEL_MAPPINGS[model_size][WHISPER_BACKEND]
    else:
        # Fallback to default if invalid model size provided
        return DEFAULT_MODELS[WHISPER_BACKEND]


# --- Helper Functions ---


def create_waveform_display(audio_data: np.ndarray, width: int = 50) -> str:
    """Create ASCII waveform visualization from audio data."""
    if len(audio_data) == 0:
        return "░" * width

    # Calculate RMS amplitude for each segment
    segment_size = max(1, len(audio_data) // width)
    amplitudes = []

    for i in range(0, len(audio_data), segment_size):
        segment = audio_data[i : i + segment_size]
        if len(segment) > 0:
            rms = np.sqrt(np.mean(segment**2))
            amplitudes.append(rms)

    if not amplitudes:
        return "░" * width

    # Normalize amplitudes to 0-4 range for visualization chars
    max_amp = max(amplitudes) if amplitudes else 1
    normalized = [
        min(4, int((amp / max_amp) * 4)) if max_amp > 0 else 0 for amp in amplitudes
    ]

    # Map to visual characters
    chars = ["░", "▒", "▓", "█", "█"]
    waveform = "".join(chars[level] for level in normalized)

    # Pad or truncate to desired width
    if len(waveform) < width:
        waveform += "░" * (width - len(waveform))
    elif len(waveform) > width:
        waveform = waveform[:width]

    return waveform


def create_frequency_bars(
    audio_data: np.ndarray, num_bars: int = 20, height: int = 8
) -> list[str]:
    """Create Winamp-style frequency spectrum bars using FFT."""
    if len(audio_data) == 0:
        return ["░" * num_bars for _ in range(height)]

    # Apply FFT to get frequency spectrum
    fft = np.abs(np.fft.rfft(audio_data))

    # Group frequencies into bars (logarithmic spacing for better visual)
    freqs_per_bar = len(fft) // num_bars
    bar_magnitudes = []

    for i in range(num_bars):
        start_idx = i * freqs_per_bar
        end_idx = min((i + 1) * freqs_per_bar, len(fft))
        if start_idx < len(fft):
            # Take average magnitude for this frequency range
            bar_mag = np.mean(fft[start_idx:end_idx])
            bar_magnitudes.append(bar_mag)
        else:
            bar_magnitudes.append(0)

    # Normalize to 0-height range
    if bar_magnitudes and max(bar_magnitudes) > 0:
        max_mag = max(bar_magnitudes)
        normalized_bars = [
            min(height, int((mag / max_mag) * height)) for mag in bar_magnitudes
        ]
    else:
        normalized_bars = [0] * num_bars

    # Create vertical bars display
    display_lines = []
    for row in range(height):
        line = ""
        for bar_height in normalized_bars:
            # Fill from bottom up (height-1-row gives us bottom-to-top)
            if bar_height > (height - 1 - row):
                line += "█"
            else:
                line += "░"
        display_lines.append(line)

    return display_lines


def create_spectrum_display(audio_data: np.ndarray, width: int = 50) -> str:
    """Create horizontal spectrum analyzer display."""
    if len(audio_data) == 0:
        return "░" * width

    # Apply FFT
    fft = np.abs(np.fft.rfft(audio_data))

    # Group into frequency bands
    freqs_per_band = len(fft) // width
    band_magnitudes = []

    for i in range(width):
        start_idx = i * freqs_per_band
        end_idx = min((i + 1) * freqs_per_band, len(fft))
        if start_idx < len(fft):
            band_mag = np.mean(fft[start_idx:end_idx])
            band_magnitudes.append(band_mag)
        else:
            band_magnitudes.append(0)

    # Normalize to 0-4 range
    if band_magnitudes and max(band_magnitudes) > 0:
        max_mag = max(band_magnitudes)
        normalized = [min(4, int((mag / max_mag) * 4)) for mag in band_magnitudes]
    else:
        normalized = [0] * width

    # Map to characters
    chars = ["░", "▒", "▓", "█", "█"]
    return "".join(chars[level] for level in normalized)


def create_vu_meter(audio_data: np.ndarray, width: int = 30) -> str:
    """Create classic VU meter display."""
    if len(audio_data) == 0:
        return "░" * width

    # Calculate RMS level
    rms = np.sqrt(np.mean(audio_data**2))

    # More sensitive scaling - amplify the signal for better visibility
    # Apply logarithmic-like scaling for natural audio response
    if rms > 0:
        # Scale and amplify for microphone input levels
        level = min(width, int((rms * width * 200) ** 0.7))
    else:
        level = 0

    # Create meter with different chars for different levels
    meter = ""
    for i in range(width):
        if i < level * 0.6:  # Green zone (safe levels)
            meter += "█"
        elif i < level * 0.85:  # Yellow zone (moderate levels)
            meter += "▓"
        elif i < level:  # Red zone (high levels)
            meter += "▒"
        else:
            meter += "░"

    return meter


def record_audio(
    filename: str,
    immediate: bool = False,
    visualize: bool = True,
    viz_style: str = "vu",
):
    if not immediate:
        print("Press [Enter] to start recording...")
        input()
    if visualize:
        print("🔴 Recording... Press [Enter] to stop.")
        if viz_style == "bars":
            # Reserve multiple lines for frequency bars
            for _ in range(5):
                print()
        else:
            print()  # Single line for other visualizations
    else:
        print("🔴 Recording... Press [Enter] to stop.")

    samplerate = 16000  # Whisper models expect 16kHz audio
    channels = 1
    dtype = "float32"
    blocksize = 1024  # Smaller blocksize for responsiveness

    stop_recording_flag = threading.Event()

    def input_listener():
        input()  # Wait for the second Enter to stop recording
        stop_recording_flag.set()

    input_thread = threading.Thread(target=input_listener)
    input_thread.start()

    frames = []
    recent_frames = []  # Keep recent frames for visualization
    max_recent_frames = 50  # About 1 second of audio for visualization

    try:
        with sd.InputStream(
            samplerate=samplerate, channels=channels, dtype=dtype, blocksize=blocksize
        ) as sd_stream:
            while not stop_recording_flag.is_set():  # Loop until the flag is set
                try:
                    data, overflowed = sd_stream.read(
                        blocksize
                    )  # Read blocksize frames
                    frames.append(data)

                    # Update visualization
                    if visualize:
                        recent_frames.append(data)
                        if len(recent_frames) > max_recent_frames:
                            recent_frames.pop(0)

                        # Create visualization from recent audio
                        if recent_frames:
                            recent_audio = np.concatenate(
                                recent_frames, axis=0
                            ).flatten()

                            if viz_style == "bars":
                                # Frequency bars (Winamp-style)
                                bar_lines = create_frequency_bars(
                                    recent_audio, num_bars=40, height=5
                                )
                                # Move cursor up 5 lines and redraw all bars
                                sys.stdout.write(f"\033[5A")  # Move up 5 lines
                                for line in bar_lines:
                                    sys.stdout.write(
                                        f"\033[K{line}\n"
                                    )  # Clear line and write
                                sys.stdout.flush()

                            elif viz_style == "spectrum":
                                # Horizontal spectrum analyzer
                                spectrum = create_spectrum_display(
                                    recent_audio, width=60
                                )
                                sys.stdout.write(f"\033[A\033[K{spectrum}\n")
                                sys.stdout.flush()

                            elif viz_style == "vu":
                                # VU meter
                                vu_meter = create_vu_meter(recent_audio, width=40)
                                sys.stdout.write(f"\033[A\033[K[{vu_meter}]\n")
                                sys.stdout.flush()

                            else:  # Default: waveform
                                waveform = create_waveform_display(
                                    recent_audio, width=60
                                )
                                sys.stdout.write(f"\033[A\033[K{waveform}\n")
                                sys.stdout.flush()

                except KeyboardInterrupt:
                    stop_recording_flag.set()
                    break
                except Exception as e:
                    print(f"Error during recording: {e}")
                    stop_recording_flag.set()
                    break
    except Exception as e:
        print(f"Error setting up recording: {e}")
        exit(1)
    finally:
        # Ensure the input thread is joined properly
        input_thread.join()  # Wait for the input thread to finish

        # Clear the visualization lines if visualization was enabled
        if visualize:
            if viz_style == "bars":
                # Clear 5 lines for frequency bars
                for _ in range(5):
                    sys.stdout.write("\033[A\033[K")
            else:
                # Clear single line for other visualizations
                sys.stdout.write("\033[A\033[K")
            sys.stdout.flush()

    if frames:
        audio_data = np.concatenate(frames, axis=0)
        sf.write(filename, audio_data, samplerate)
        print("✅ Recording finished.")
    else:
        print("⚠️ No audio recorded.")
        exit(1)


def get_default_output_folder() -> str | None:
    """Get the default output folder from environment variable."""
    return os.environ.get("WHISPER_OUTPUT_FOLDER")


def determine_output_path(path_arg: str | None) -> str:
    default_prefix = "transcription"
    default_folder = get_default_output_folder()

    if not path_arg:
        # No path specified - use default folder if set, otherwise current directory
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{default_prefix}_{timestamp}.txt"

        if default_folder:
            os.makedirs(default_folder, exist_ok=True)
            return os.path.join(default_folder, filename)
        else:
            return f"./{filename}"
    elif path_arg.endswith(os.sep):
        # Directory specified
        os.makedirs(path_arg, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return os.path.join(path_arg, f"{default_prefix}_{timestamp}.txt")
    else:
        # Specific file path specified
        output_dir = os.path.dirname(path_arg)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return path_arg


def edit_transcription(text: str) -> str:
    print("📝 Transcription complete. Opening in $EDITOR for review...")
    editor = os.environ.get("EDITOR", "nano")

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(text)
        tmp_file_path = tmp_file.name

    try:
        subprocess.run([editor, tmp_file_path], check=True)
        with open(tmp_file_path, "r", encoding="utf-8") as tmp_file_read:
            edited_text = tmp_file_read.read()
        return edited_text
    except subprocess.CalledProcessError as e:
        print(f"Error opening editor: {e}")
        return text  # Return original text if editor fails
    finally:
        os.remove(tmp_file_path)


# --- Main Execution ---


def main():
    parser = argparse.ArgumentParser(
        description="Record audio, transcribe it, and save to a file."
    )
    parser.add_argument(
        "-e",
        "--edit",
        action="store_true",
        help="Open the transcribed text in $EDITOR before saving.",
    )
    parser.add_argument(
        "-i",
        "--immediate",
        action="store_true",
        help="Start recording immediately without waiting for user input.",
    )
    default_model = get_default_model()
    parser.add_argument(
        "-m",
        "--model",
        default=default_model,
        choices=list(MODEL_MAPPINGS.keys()),
        help=f"Whisper model size to use (default: {default_model}). Options: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, turbo, large-v3-turbo. Can be overridden with WHISPER_DEFAULT_MODEL env var.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable real-time audio visualization during recording.",
    )
    parser.add_argument(
        "--viz-style",
        choices=["waveform", "bars", "spectrum", "vu"],
        default="vu",
        help="Visualization style: vu (VU meter, default), waveform, bars (Winamp-style frequency), or spectrum (horizontal).",
    )
    default_folder = get_default_output_folder()
    folder_help = (
        f" Defaults to WHISPER_OUTPUT_FOLDER env var ({default_folder}) if set."
        if default_folder
        else " Can be set via WHISPER_OUTPUT_FOLDER env var."
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help=f"Optional: Path to save the transcription. Can be a file or a directory (ending with /).{folder_help}",
    )

    args = parser.parse_args()

    # Get the model name for the selected size
    model_name = get_model_name(args.model)

    # 1. Record Audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
        audio_file_path = audio_tmp.name
    record_audio(audio_file_path, args.immediate, not args.no_visualize, args.viz_style)

    # 2. Transcribe Audio
    print(
        f"🎤 Transcribing audio using {WHISPER_BACKEND} with {args.model} model... (This may take a moment)"
    )
    try:
        if WHISPER_BACKEND == "mlx":
            # MLX Whisper API
            result = whisper_lib(audio_file_path, path_or_hf_repo=model_name)
            transcribed_text = result["text"]
        else:
            # OpenAI Whisper API
            model = whisper_lib.load_model(model_name)
            result = model.transcribe(audio_file_path)
            transcribed_text = result["text"]
    except Exception as e:
        print(f"⚠️ Transcription failed: {e}")
        transcribed_text = ""  # Ensure it's empty if transcription fails

    os.remove(audio_file_path)  # Clean up temporary audio file

    if not transcribed_text:
        print("⚠️ Transcription produced no text. Exiting.")
        exit(1)

    # 3. Optional Edit Step
    if args.edit:
        final_text = edit_transcription(transcribed_text)
    else:
        final_text = transcribed_text

    # 4. Determine Output Path and Save
    output_file_path = determine_output_path(args.output_path)
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"✅ Saved to: {output_file_path}")


if __name__ == "__main__":
    main()

