# app.py Documentation

## Overview
`app.py` is a full-featured Gradio web application that transcribes Chinese audio files and optionally translates them to English. It combines ModelScope's Paraformer ASR model with Alibaba's Qwen translation API (via OpenAI-compatible interface).

## Features

### 1. Multi-Format Audio Support
- **Supported formats**: MP3, WAV, and other audio formats supported by pydub
- **Smart WAV detection**: If input is already WAV format, skips unnecessary conversion
- **Format standardization**: All audio is converted to 16kHz, mono WAV for optimal ASR performance

### 2. Chinese Speech Recognition
- **Model**: `speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` from ModelScope
- **Device support**: Automatically uses GPU if available, falls back to CPU
- **Robust output handling**: Handles both list and dict output formats from the ASR pipeline

### 3. Optional English Translation
- **Translation API**: Qwen Plus model via DashScope OpenAI-compatible API
- **User control**: Checkbox to enable/disable translation (saves API costs when not needed)
- **Smart token management**: Automatically estimates required tokens based on input length
- **Usage tracking**: Displays prompt, completion, and total token usage

### 4. Markdown Report Generation
- **Automatic export**: Creates timestamped `.md` files with results
- **Conditional sections**: Only includes translation section if translation was performed
- **Metadata**: Includes audio filename, generation timestamp, and models used

## Application Flow

```
1. User uploads audio file (MP3, WAV, etc.)
   ↓
2. Audio format detection and conversion
   - If WAV: Ensure 16kHz mono format
   - If other: Convert to 16kHz mono WAV
   ↓
3. Chinese transcription using Paraformer ASR
   ↓
4. Optional translation (if checkbox enabled)
   - Translate Chinese → English using Qwen API
   ↓
5. Generate markdown report
   - Include Chinese transcript
   - Include English translation (if performed)
   ↓
6. Display results and provide downloadable MD file
```

## Key Functions

### `convert_audio_to_wav(audio_path, wav_path)`
Converts audio files to standardized WAV format (16kHz, mono).
- **Smart detection**: Checks if input is already WAV
- **Format enforcement**: Always ensures correct sample rate and channels
- **Error handling**: Raises RuntimeError if conversion fails

### `transcribe_audio(wav_path)`
Transcribes Chinese speech to text using ModelScope Paraformer.
- **Returns**: Chinese text string
- **Fallback**: Returns error message if no speech detected
- **Console output**: Prints full transcript for logging

### `translate_chinese_to_english_openai(chinese_text)`
Translates Chinese text to English using Qwen API.
- **Token estimation**: Calculates required output tokens (max 2048)
- **API parameters**:
  - Model: `qwen-plus`
  - Temperature: 0.7
  - Top_p: 0.9
- **Error handling**: Returns error message if translation fails

### `generate_markdown_file(audio_filename, chinese_text, english_translation=None)`
Creates a formatted markdown report.
- **Timestamped filenames**: `transcript_<timestamp>.md`
- **Conditional formatting**: Only adds translation section if provided
- **UTF-8 encoding**: Proper handling of Chinese characters

### `process_audio_file(audio_file, need_translation=True)`
Main orchestrator function for Gradio interface.
- **Coordinates**: All processing steps
- **Cleanup**: Removes temporary WAV files
- **Error handling**: Catches and reports all exceptions

## Gradio Interface

### Inputs
- **Audio File Upload**: Accepts any audio format
- **Translation Checkbox**: Toggle English translation (default: enabled)

### Outputs
- **Chinese Transcript**: Text display of transcribed Chinese
- **English Translation**: Text display of translated English (if enabled)
- **Markdown Report**: Downloadable `.md` file with complete results

### UI Elements
- Title: "🎙️ Chinese Audio to English Translator"
- Button: "🚀 Transcribe & Translate"
- Info text explaining functionality and API usage

## Environment Requirements

### Environment Variables
```
DASHSCOPE_API_KEY=your_api_key_here
```
Required for Qwen translation API access. Get your key at:
https://dashscope-intl.aliyuncs.com/compatible-mode/v1

### Dependencies
- `gradio`: Web interface
- `modelscope`: ASR pipeline
- `pydub`: Audio processing
- `openai`: API client for Qwen
- `python-dotenv`: Environment variable management
- `torch`: Deep learning framework
- `ffmpeg`: Audio codec (system dependency)

### System Requirements
- **ffmpeg**: Must be installed and accessible in PATH
- **GPU** (optional): CUDA-compatible GPU for faster ASR processing

## Usage Example

### Basic Usage
1. Start the application:
```bash
python app.py
```

2. Open the Gradio interface in your browser
3. Upload a Chinese audio file (MP3, WAV, etc.)
4. Check/uncheck "Translate to English" as needed
5. Click "🚀 Transcribe & Translate"
6. View results and download markdown report

### Output Example
**Console output:**
```
============================================================
Starting audio processing...
============================================================
✅ File is already WAV format: sample.wav
✅ Ensured correct format (16kHz, mono): ./temp_input.wav
Using device: gpu
✅ Chinese Transcript: 你好，欢迎使用语音转文字系统...
Translating 156 chars (max_tokens: 203)...
✅ Translation complete:
   - Prompt tokens: 189
   - Completion tokens: 198
   - Total tokens: 387
✅ Markdown file saved: transcript_1733425678.md
============================================================
✅ Processing completed successfully!
============================================================
```

**Generated MD file:**
```markdown
# Audio Transcription Report

**Audio File**: `sample.wav`
**Generated On**: 2024-12-05 14:21:18
**Model Used**:
- ASR: `speech_paraformer-large` (ModelScope)
- Translation: `qwen-plus` (DashScope API via OpenAI Wrapper)

---

## 🇨🇳 Chinese Transcript
你好，欢迎使用语音转文字系统...

---

## 🇬🇧 English Translation
Hello, welcome to the speech-to-text system...

---
*Generated by Qwen + ModelScope Pipeline*
```

## Cost Optimization

### Translation Toggle
The translation checkbox allows users to skip translation when not needed:
- ✅ **Checked**: Transcribe + Translate (uses API tokens)
- ❌ **Unchecked**: Transcribe only (no API costs)

This is useful when:
- You only need Chinese transcripts
- Processing multiple files in batches
- Testing/debugging transcription accuracy

## Error Handling

### Conversion Errors
```python
RuntimeError: Failed to convert audio: [detailed error message]
```

### Transcription Errors
```python
"(No speech detected or transcription failed.)"
```

### Translation Errors
```python
"Translation failed: [API error details]"
```

### File Upload Errors
```python
"Please upload an audio file."
```

## Technical Notes

### Audio Processing
- Temporary WAV files are created in the working directory
- Files are automatically cleaned up after processing
- Original audio files are never modified

### API Limits
- Max tokens per translation: 2048
- Token estimation: `max(512, len(chinese_text) * 1.3)`
- Model: `qwen-plus` (configurable to `qwen-turbo` or `qwen-max`)

### Device Selection
```python
device = 'gpu' if torch.cuda.is_available() else 'cpu'
```
Automatically uses GPU for faster ASR processing when available.

## Customization Options

### Change Translation Model
Line 144: Modify `model` parameter
```python
model="qwen-plus"  # Options: qwen-turbo, qwen-plus, qwen-max
```

### Adjust Translation Parameters
Lines 151-153:
```python
temperature=0.7,  # Creativity (0.0-1.0)
top_p=0.9,       # Nucleus sampling
max_tokens=actual_max  # Output length limit
```

### Enable Public Sharing
Line 324:
```python
demo.launch(share=True)  # Creates public URL
```

## File Structure
```
chineseTranscribe/
├── app.py                    # Main application
├── .env                      # API keys (not tracked)
├── .gitignore               # Git ignore rules
├── transcript_*.md          # Generated reports (not tracked)
├── temp_input.wav           # Temporary file (auto-deleted)
└── app_documentation.md     # This file
```

## Powered By
- **ASR**: ModelScope Paraformer (Alibaba DAMO Academy)
- **Translation**: Qwen API (Alibaba Cloud)
- **Interface**: Gradio
- **Audio Processing**: pydub + ffmpeg