# Whisper Model Comparison Guide

This guide helps you choose the right Whisper model for your transcription needs, balancing accuracy, speed, and resource usage.

## Quick Model Selection

### For Different Use Cases:

- **Quick Notes/Drafts**: `tiny` or `tiny.en` (fastest, basic accuracy)
- **Fast Transcription**: `small.en` (good balance for older systems)
- **General Note-taking**: `large-v3-turbo` (default - best balance)
- **Alternative Quality**: `turbo` (also excellent)
- **Resource-Limited Systems**: `tiny.en` or `base.en`

## Model Comparison Chart

| Model | Parameters | VRAM Usage | Relative Speed | English WER | Best For | Memory Efficient |
|-------|------------|------------|----------------|-------------|-----------|-----------------|
| **tiny** | 39M | ~1 GB | 32x | Higher | Quick drafts | ✅ Excellent |
| **tiny.en** | 39M | ~1 GB | 32x | Good | Fast English-only | ✅ Excellent |
| **base** | 74M | ~1 GB | 16x | Moderate | Basic transcription | ✅ Very Good |
| **base.en** | 74M | ~1 GB | 16x | Good | English transcription | ✅ Very Good |
| **small** | 244M | ~2 GB | 6x | Good | Balanced multilingual | ✅ Good |
| **small.en** | 244M | ~2 GB | 6x | Very Good | Fast transcription | ✅ Good |
| **turbo** | 809M | ~3 GB | 8x | Very Good | Alternative quality | ⚠️ Moderate |
| **medium** | 769M | ~5 GB | 2x | Very Good | High accuracy | ❌ High |
| **medium.en** | 769M | ~5 GB | 2x | Excellent | English high accuracy | ❌ High |
| **large-v3-turbo** | 809M | ~3 GB | 8x | Excellent | **Default choice** | ⚠️ Moderate |

## Performance Details

### Speed Comparison
Based on research from multiple sources, speed is relative to the large model (1x baseline):

- **tiny/tiny.en**: 32x faster than large
- **base/base.en**: 16x faster than large  
- **small/small.en**: 6x faster than large
- **turbo/large-v3-turbo**: 8x faster than large
- **medium/medium.en**: 2x faster than large

### Accuracy Analysis
English Word Error Rate (WER) - lower is better:

- **Large models**: ~3-5% WER (best accuracy)
- **Turbo models**: ~3-6% WER (near-large accuracy)  
- **Medium models**: ~5-8% WER (very good)
- **Small models**: ~6-10% WER (good)
- **Base models**: ~8-12% WER (moderate)
- **Tiny models**: ~12-20% WER (basic)

*Note: Human transcribers typically achieve 3-7% WER on the same datasets*

### Memory Requirements

#### VRAM (GPU Memory):
- **tiny/base**: ~1 GB VRAM
- **small**: ~2 GB VRAM  
- **turbo/large-v3-turbo**: ~3 GB VRAM
- **medium**: ~5 GB VRAM
- **large**: ~10 GB VRAM

#### System RAM:
- All models require additional system RAM equal to or greater than VRAM usage
- Recommended: 2x the VRAM requirement in system RAM for optimal performance

## Usage Examples

### Basic Usage
```bash
# Use default model (large-v3-turbo) - best balance
make run

# Quick transcription with tiny model
make run ARGS="-m tiny.en"

# Fast transcription for older systems
make run ARGS="-m small.en"

# Alternative high-quality option
make run ARGS="-m turbo"
```

### Advanced Usage
```bash
# Transcribe to specific file (uses default high-quality model)
make run ARGS="meeting-notes.txt"

# Edit transcription before saving (uses default model)
make run ARGS="-e"

# Save to directory with timestamp (uses default model)
make run ARGS="notes/"

# Use faster model for quick notes
make run ARGS="-m small.en -e notes/quick-$(date +%Y%m%d).txt"
```

### Direct Python Usage
```bash
# Using uv directly (default model)
uv run python voice2txt.py
uv run python voice2txt.py -e
uv run python voice2txt.py important-meeting.txt

# Using specific models
uv run python voice2txt.py -m small.en -e
uv run python voice2txt.py -m turbo notes/

# Using environment variables to customize defaults
WHISPER_DEFAULT_MODEL=small.en uv run python voice2txt.py
WHISPER_OUTPUT_FOLDER=~/notes uv run python voice2txt.py

# Persistent settings
export WHISPER_DEFAULT_MODEL=tiny.en
export WHISPER_OUTPUT_FOLDER=~/voice-transcriptions
```

## Model Recommendations by Hardware

### Laptop/Desktop (8-16 GB RAM):
1. **Recommended**: `large-v3-turbo` (default - excellent balance)
2. **Alternative**: `small.en` for older/slower systems
3. **Avoid**: `medium` or larger non-turbo models

### High-End Desktop (32+ GB RAM):
1. **Recommended**: `large-v3-turbo` (default - optimal choice)
2. **Alternative**: `medium` for slightly better accuracy
3. **Note**: Default model already provides excellent quality

### Resource-Constrained Systems:
1. **Recommended**: `tiny.en` or `base.en`
2. **Trade-off**: Accept lower accuracy for speed
3. **Strategy**: Use for drafts, edit manually
4. **Environment Setup**: 
   ```bash
   export WHISPER_DEFAULT_MODEL=tiny.en      # Faster model
   export WHISPER_OUTPUT_FOLDER=~/quick-notes # Organized storage
   ```

## Language-Specific Considerations

### English-Only Content:
- **Default choice**: `large-v3-turbo` (multilingual but excellent for English)
- **Faster alternatives**: `small.en`, `base.en` (10-15% better than multilingual equivalents)
- **Note**: Default model performs excellently on English despite being multilingual

### Multilingual Content:
- **Default choice**: `large-v3-turbo` (excellent multilingual support)
- **Alternatives**: `small`, `medium`, `turbo`
- **Note**: Default model handles accents and multiple languages very well

## Environment Variables Configuration

The voice2txt utility supports two environment variables for customizing defaults:

### `WHISPER_DEFAULT_MODEL`
Set your preferred model to avoid typing `-m` every time:

```bash
# For users with older systems (faster, lower memory)
export WHISPER_DEFAULT_MODEL=small.en

# For maximum speed on resource-limited systems  
export WHISPER_DEFAULT_MODEL=tiny.en

# For alternative quality (if you prefer over large-v3-turbo)
export WHISPER_DEFAULT_MODEL=turbo
```

### `WHISPER_OUTPUT_FOLDER`
Set a default folder for organized transcription storage:

```bash
# Personal notes organization
export WHISPER_OUTPUT_FOLDER=~/voice-notes

# Work-related transcriptions
export WHISPER_OUTPUT_FOLDER=~/Documents/Meetings

# Project-specific folder
export WHISPER_OUTPUT_FOLDER=./transcriptions
```

### Combined Workflow Examples

```bash
# Setup for quick personal notes
export WHISPER_DEFAULT_MODEL=small.en
export WHISPER_OUTPUT_FOLDER=~/quick-notes
# Now just run: make run

# Setup for high-quality meeting transcriptions  
export WHISPER_DEFAULT_MODEL=large-v3-turbo  # (default anyway)
export WHISPER_OUTPUT_FOLDER=~/Documents/Meetings
# Now just run: make run ARGS="-e"

# Setup for development/testing (fast iterations)
export WHISPER_DEFAULT_MODEL=tiny.en
export WHISPER_OUTPUT_FOLDER=./test-transcriptions
```

### Precedence Rules

1. **Command-line arguments** always override environment variables
2. **Environment variables** override built-in defaults
3. **Built-in defaults** are used when nothing else is specified

```bash
# Environment setup
export WHISPER_DEFAULT_MODEL=small.en
export WHISPER_OUTPUT_FOLDER=~/notes

# These use environment variables:
make run                           # Uses small.en, saves to ~/notes/
make run ARGS="-e"                # Uses small.en, saves to ~/notes/

# These override environment variables:
make run ARGS="-m turbo"          # Uses turbo, saves to ~/notes/
make run ARGS="meeting.txt"       # Uses small.en, saves to ./meeting.txt
make run ARGS="-m turbo custom/"  # Uses turbo, saves to custom/
```

## Technical Notes

### MLX vs OpenAI Whisper Backend
- **macOS with Apple Silicon**: Automatically uses MLX backend for optimal performance
- **Other platforms**: Uses OpenAI Whisper backend
- **Performance**: MLX backend typically 2-4x faster on Apple Silicon
- **Compatibility**: Model selection works identically across both backends

### Model Quantization (MLX)
Some MLX models use quantization for reduced memory:
- **4-bit quantized**: Smaller memory footprint, minimal accuracy loss
- **8-bit models**: Available for most sizes
- **FP32 models**: Full precision, larger memory usage

## Model Downloads & Authentication

### Hugging Face Account Requirements

**✅ No Account Required**: Most models work without Hugging Face authentication
- All MLX-community models are publicly available
- Models download automatically on first use
- No login or token needed for standard usage

**⚠️ Account May Be Required For**:
- Some newer/experimental models
- Models with download restrictions
- Rate-limited access during high usage periods

### Setting Up Authentication (Optional)

If you encounter authentication errors, you can optionally set up a Hugging Face account:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login (optional - only if needed)
huggingface-cli login

# Or set token as environment variable
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

## Troubleshooting

### Common Issues:

**Model Download Errors:**
- Ensure internet connection for first-time model downloads
- MLX models download from Hugging Face automatically  
- Large models may take several minutes to download
- **If download fails**: Try again later or use a smaller model

**Memory Errors:**
- Reduce model size if running out of VRAM/RAM
- Close other applications to free memory
- Consider quantized models for lower memory usage
- **Quick fix**: `export WHISPER_DEFAULT_MODEL=small.en` for lower memory usage

**Slow Performance:**
- Use smaller models for faster transcription
- Ensure sufficient system resources
- Consider `turbo` models for speed/accuracy balance
- **Quick fix**: `export WHISPER_DEFAULT_MODEL=tiny.en` for maximum speed

**Environment Variable Issues:**
- **Model variable**: Ensure name is exactly `WHISPER_DEFAULT_MODEL`
- **Folder variable**: Ensure name is exactly `WHISPER_OUTPUT_FOLDER` 
- Use valid model names (see choices in help output)
- Invalid models automatically fallback to `large-v3-turbo`
- Folder paths are automatically created if they don't exist

## Sources and References

1. **OpenAI Whisper Paper**: "Robust Speech Recognition via Large-Scale Weak Supervision" 
   - https://cdn.openai.com/papers/whisper.pdf
   - Original benchmarks and model specifications

2. **Speechly Analysis**: "Analyzing OpenAI's Whisper ASR Accuracy: Word Error Rates Across Languages and Model Sizes"
   - https://www.speechly.com/blog/analyzing-open-ais-whisper-asr-models-word-error-rates-across-languages
   - Comprehensive WER analysis across languages

3. **Zeno Hub Comparison**: "Comparing OpenAI Whisper Transcription Models"
   - https://hub.zenoml.com/report/1123/Comparing%20OpenAI%20Whisper%20Transcription%20Models
   - Performance benchmarks and model comparisons

4. **MLX Community Models**: Hugging Face MLX-Community Collection
   - https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
   - MLX-optimized model implementations

5. **Tom's Hardware GPU Benchmarks**: "OpenAI Whisper Audio Transcription Benchmarked on 18 GPUs"
   - https://www.tomshardware.com/news/whisper-audio-transcription-gpus-benchmarked
   - GPU performance and memory usage data

6. **Towards AI**: "Whisper Variants Comparison: Features and Implementation"
   - https://towardsai.net/p/machine-learning/whisper-variants-comparison-what-are-their-features-and-how-to-implement-them
   - Model variant analysis and recommendations

---

*Last updated: 2024-08-20*
*For the most current model availability, check: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc*