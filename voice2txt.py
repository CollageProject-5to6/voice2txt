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
import queue
import collections

# Optional memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


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


class StreamingAudioCapture:
    """Manages streaming audio capture with circular buffer for continuous transcription."""
    
    def __init__(self, chunk_size: int = 10, buffer_size: int = 30, samplerate: int = 16000):
        self.chunk_size = chunk_size  # seconds
        self.buffer_size = buffer_size  # seconds  
        self.samplerate = samplerate
        self.channels = 1
        self.dtype = "float32"
        self.blocksize = 1024
        
        # Circular buffer to store recent audio
        buffer_frames = buffer_size * samplerate
        self.audio_buffer = collections.deque(maxlen=buffer_frames)
        
        # Thread coordination
        self.stop_event = threading.Event()
        self.chunk_queue = queue.Queue()
        self.audio_thread = None
        self.transcription_thread = None
        
        # Visualization state
        self.recent_frames = []
        self.max_recent_frames = 50
        
        # Memory monitoring
        self.memory_check_interval = 60  # seconds
        self.last_memory_check = 0
        
    def start_capture(self, visualize: bool = True, viz_style: str = "vu"):
        """Start the streaming audio capture."""
        self.visualize = visualize
        self.viz_style = viz_style
        
        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self._audio_capture_loop)
        self.audio_thread.start()
        
        # Start chunk processing thread
        self.transcription_thread = threading.Thread(target=self._chunk_processor_loop)
        self.transcription_thread.start()
        
    def stop_capture(self):
        """Stop the streaming audio capture."""
        self.stop_event.set()
        if self.audio_thread:
            self.audio_thread.join()
        if self.transcription_thread:
            self.transcription_thread.join()
            
    def _audio_capture_loop(self):
        """Main audio capture loop running in separate thread."""
        try:
            with sd.InputStream(
                samplerate=self.samplerate, 
                channels=self.channels, 
                dtype=self.dtype, 
                blocksize=self.blocksize
            ) as stream:
                while not self.stop_event.is_set():
                    try:
                        data, overflowed = stream.read(self.blocksize)
                        
                        # Add to circular buffer
                        for sample in data.flatten():
                            self.audio_buffer.append(sample)
                            
                        # Update visualization
                        if self.visualize:
                            self._update_visualization(data)
                            
                    except Exception as e:
                        print(f"⚠️ Audio capture error: {e}")
                        break
                        
        except Exception as e:
            print(f"⚠️ Failed to start audio stream: {e}")
            
    def _update_visualization(self, data):
        """Update real-time visualization."""
        self.recent_frames.append(data)
        if len(self.recent_frames) > self.max_recent_frames:
            self.recent_frames.pop(0)
            
        if self.recent_frames:
            recent_audio = np.concatenate(self.recent_frames, axis=0).flatten()
            
            if self.viz_style == "bars":
                bar_lines = create_frequency_bars(recent_audio, num_bars=40, height=5)
                sys.stdout.write(f"\033[5A")
                for line in bar_lines:
                    sys.stdout.write(f"\033[K{line}\n")
                sys.stdout.flush()
            elif self.viz_style == "spectrum":
                spectrum = create_spectrum_display(recent_audio, width=60)
                sys.stdout.write(f"\033[A\033[K{spectrum}\n")
                sys.stdout.flush()
            elif self.viz_style == "vu":
                vu_meter = create_vu_meter(recent_audio, width=40)
                sys.stdout.write(f"\033[A\033[K[{vu_meter}]\n")
                sys.stdout.flush()
            else:  # waveform
                waveform = create_waveform_display(recent_audio, width=60)
                sys.stdout.write(f"\033[A\033[K{waveform}\n")
                sys.stdout.flush()
                
    def _chunk_processor_loop(self):
        """Process audio chunks for transcription."""
        chunk_frames = self.chunk_size * self.samplerate
        last_chunk_time = time.time()
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            # Process chunk every chunk_size seconds
            if current_time - last_chunk_time >= self.chunk_size:
                if len(self.audio_buffer) >= chunk_frames:
                    # Extract chunk from buffer
                    chunk_data = np.array(list(self.audio_buffer)[-chunk_frames:])
                    
                    # Put chunk in queue for transcription
                    timestamp = datetime.datetime.now()
                    self.chunk_queue.put((chunk_data, timestamp))
                    
                    last_chunk_time = current_time
                    
                # Memory monitoring
                current_time = time.time()
                if current_time - self.last_memory_check > self.memory_check_interval:
                    self._check_memory_usage()
                    self.last_memory_check = current_time
                    
            # Small sleep to prevent busy waiting
            time.sleep(0.1)
            
    def get_chunk(self, timeout: float = 1.0):
        """Get the next audio chunk for transcription."""
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
            
    def _check_memory_usage(self):
        """Monitor memory usage and log warnings if necessary."""
        if not PSUTIL_AVAILABLE:
            return
            
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Warn if memory usage exceeds 500MB
            if memory_mb > 500:
                print(f"⚠️  High memory usage: {memory_mb:.1f}MB")
                
            # Force garbage collection if over 200MB
            if memory_mb > 200:
                import gc
                gc.collect()
                
        except Exception as e:
            # Don't fail on memory monitoring errors
            pass


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


def stream_transcribe(
    output_path: str,
    model_name: str,
    chunk_size: int = 10,
    buffer_size: int = 30,
    immediate: bool = False,
    visualize: bool = True,
    viz_style: str = "vu"
):
    """Main streaming transcription function."""
    print(f"🎤 Starting streaming transcription with {WHISPER_BACKEND} ({model_name})")
    print(f"📝 Output file: {output_path}")
    print(f"⚙️  Chunk size: {chunk_size}s, Buffer size: {buffer_size}s")
    print()
    
    if not immediate:
        print("Press [Enter] to start streaming...")
        input()
    
    # Initialize audio capture
    capture = StreamingAudioCapture(
        chunk_size=chunk_size,
        buffer_size=buffer_size
    )
    
    # Load Whisper model
    print("🔄 Loading Whisper model...")
    if WHISPER_BACKEND == "mlx":
        # MLX doesn't need pre-loading
        whisper_model = None
    else:
        whisper_model = whisper_lib.load_model(model_name)
    
    # Setup visualization
    if visualize:
        print("🔴 Streaming... Press [Enter] to stop.")
        if viz_style == "bars":
            for _ in range(5):
                print()
        else:
            print()
    else:
        print("🔴 Streaming... Press [Enter] to stop.")
    
    # Start capture
    capture.start_capture(visualize=visualize, viz_style=viz_style)
    
    # Setup output file
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(f"# Streaming Transcription - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        output_file.flush()
        
        # Setup stop listener
        stop_flag = threading.Event()
        
        def input_listener():
            input()
            stop_flag.set()
            
        input_thread = threading.Thread(target=input_listener)
        input_thread.start()
        
        # Main transcription loop
        chunk_count = 0
        last_text = ""
        
        try:
            while not stop_flag.is_set():
                # Get next audio chunk
                chunk_data, timestamp = capture.get_chunk(timeout=1.0)
                
                if chunk_data is not None:
                    chunk_count += 1
                    
                    # Create temporary file for this chunk
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        sf.write(temp_audio.name, chunk_data, capture.samplerate)
                        temp_audio_path = temp_audio.name
                    
                    try:
                        # Transcribe chunk
                        if WHISPER_BACKEND == "mlx":
                            result = whisper_lib(temp_audio_path, path_or_hf_repo=model_name)
                            text = result["text"].strip()
                        else:
                            result = whisper_model.transcribe(temp_audio_path)
                            text = result["text"].strip()
                        
                        # Only write if we have new content
                        if text and text != last_text:
                            # Simple deduplication - avoid repeating the same text
                            if not last_text or not text.startswith(last_text[:50]):
                                datetime_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                                output_line = f"[{datetime_str}] {text}\n"
                                output_file.write(output_line)
                                output_file.flush()
                                
                                # Show progress (clear visualization and show text)
                                if visualize:
                                    if viz_style == "bars":
                                        sys.stdout.write(f"\033[5A")
                                        for _ in range(5):
                                            sys.stdout.write(f"\033[K\n")
                                        sys.stdout.write(f"\033[5A")
                                    else:
                                        sys.stdout.write(f"\033[A\033[K")
                                    
                                print(f"✅ [{datetime_str}] {text}")
                                
                                if visualize:
                                    if viz_style == "bars":
                                        for _ in range(5):
                                            print()
                                    else:
                                        print()
                                
                                last_text = text
                        
                    except Exception as e:
                        print(f"⚠️ Transcription error: {e}")
                    finally:
                        # Cleanup temp file
                        try:
                            os.remove(temp_audio_path)
                        except:
                            pass
                            
        except KeyboardInterrupt:
            stop_flag.set()
        finally:
            # Cleanup
            capture.stop_capture()
            input_thread.join()
            
            # Clear visualization
            if visualize:
                if viz_style == "bars":
                    for _ in range(5):
                        sys.stdout.write("\033[A\033[K")
                else:
                    sys.stdout.write("\033[A\033[K")
                sys.stdout.flush()
                
    print(f"✅ Streaming complete. Processed {chunk_count} chunks.")
    print(f"📄 Transcription saved to: {output_path}")


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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode for continuous transcription without storing full audio files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Audio chunk size in seconds for streaming mode (default: 10).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=30,
        help="Audio buffer size in seconds for streaming mode (default: 30).",
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
    
    # Determine output path
    output_file_path = determine_output_path(args.output_path)

    if args.stream:
        # Streaming mode
        if args.edit:
            print("⚠️ Edit mode not supported in streaming mode. Transcription will be saved directly.")
        
        stream_transcribe(
            output_path=output_file_path,
            model_name=model_name,
            chunk_size=args.chunk_size,
            buffer_size=args.buffer_size,
            immediate=args.immediate,
            visualize=not args.no_visualize,
            viz_style=args.viz_style
        )
    else:
        # Traditional mode (record full audio then transcribe)
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

        # 4. Save
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        print(f"✅ Saved to: {output_file_path}")


if __name__ == "__main__":
    main()

