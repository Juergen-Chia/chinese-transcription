# app.py - Full Gradio Web App with OpenAI Wrapper for Qwen API
import os
import gradio as gr
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pydub import AudioSegment
from pydub.utils import which
from openai import OpenAI  # New: OpenAI wrapper for Qwen
import markdown
import datetime
from dotenv import load_dotenv

# ========================================
# Configure pydub to use ffmpeg (must be before any AudioSegment operations)
# ========================================
AudioSegment.converter = which("ffmpeg")  # Find and set ffmpeg path
AudioSegment.ffprobe = which("ffprobe")    # Find and set ffprobe path

# ========================================
# Load Environment Variables
# ========================================
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError(
        "DASHSCOPE_API_KEY not found in .env file. "
        "Please get your key at https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )

# ========================================
# Initialize OpenAI client with DashScope base URL
# ========================================
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# ========================================
# Step 1: Convert MP3 to WAV (16kHz, mono)
# ========================================
def convert_mp3_to_wav(mp3_path, wav_path):
    """
    Convert MP3 audio file to WAV format (16kHz, mono) for ASR processing.

    Args:
        mp3_path (str): Path to input MP3 file
        wav_path (str): Path to output WAV file

    Returns:
        str: Path to converted WAV file
    """
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        print(f"✅ Converted {mp3_path} to {wav_path}")
        return wav_path
    except Exception as e:
        raise RuntimeError(f"Failed to convert MP3: {str(e)}")

# ========================================
# Step 2: Transcribe Chinese Speech → Text
# ========================================
def transcribe_audio(wav_path):
    """
    Transcribe Chinese audio to text using ModelScope Paraformer.

    Args:
        wav_path (str): Path to WAV audio file

    Returns:
        str: Transcribed Chinese text
    """
    import torch  # Import here to avoid issues if not needed

    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    asr_pipeline = pipeline(
        task='auto-speech-recognition',
        model='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        device=device
    )
    result = asr_pipeline(wav_path)

    # Handle both list and dict output formats
    if isinstance(result, list):
        chinese_text = "".join([seg["text"] for seg in result])
    elif isinstance(result, dict):
        chinese_text = result.get("text", "")
    else:
        chinese_text = str(result)  # Fallback

    chinese_text = chinese_text.strip()
    if not chinese_text:
        chinese_text = "(No speech detected or transcription failed.)"

    print(f"✅ Chinese Transcript: {chinese_text[:100]}...")
    return chinese_text

# ========================================
# Step 3: Translate Chinese → English using OpenAI Wrapper
# ========================================
def translate_chinese_to_english_openai(chinese_text):
    """
    Translate Chinese text to English using Qwen model via OpenAI-compatible API.

    Args:
        chinese_text (str): Chinese text to translate

    Returns:
        str: English translation
    """
    try:
        # Estimate required output length
        estimated_output_tokens = max(512, int(len(chinese_text) * 1.3))
        actual_max = min(estimated_output_tokens, 2048)  # Cap at safe limit

        print(f"Translating {len(chinese_text)} chars (max_tokens: {actual_max})...")

        # Call Qwen model using OpenAI-compatible interface
        completion = client.chat.completions.create(
            model="qwen-plus",  # Options: qwen-turbo, qwen-plus, qwen-max
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the following Chinese text into fluent, natural English. Be complete and do not summarize:\n\n{chinese_text}"
                }
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=actual_max
        )

        # Extract translated text
        translated = completion.choices[0].message.content.strip()

        # Display token usage info
        if completion.usage:
            print(f"✅ Translation complete:")
            print(f"   - Prompt tokens: {completion.usage.prompt_tokens}")
            print(f"   - Completion tokens: {completion.usage.completion_tokens}")
            print(f"   - Total tokens: {completion.usage.total_tokens}")

        return translated

    except Exception as e:
        error_msg = f"Translation failed: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# ========================================
# Step 4: Generate Markdown (.md) File
# ========================================
def generate_markdown_file(audio_filename, chinese_text, english_translation):
    """
    Generate a markdown report with transcription and translation.

    Args:
        audio_filename (str): Original audio filename
        chinese_text (str): Chinese transcript
        english_translation (str): English translation

    Returns:
        str: Path to generated markdown file
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content = f"""# Audio Translation Report

**Audio File**: `{audio_filename}`
**Generated On**: {timestamp}
**Model Used**:
- ASR: `speech_paraformer-large` (ModelScope)
- Translation: `qwen-plus` (DashScope API via OpenAI Wrapper)

---

## 🇨🇳 Chinese Transcript
{chinese_text}

---

## 🇬🇧 English Translation
{english_translation}

---
*Generated by Qwen + ModelScope Pipeline*
"""
    output_path = f"transcript_{int(datetime.datetime.now().timestamp())}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"✅ Markdown file saved: {output_path}")
    return output_path

# ========================================
# Gradio Interface Wrapper
# ========================================
def process_audio_file(audio_file):
    """
    Main processing function for Gradio interface.

    Args:
        audio_file: Gradio file upload (path as string)

    Returns:
        tuple: (chinese_text, english_translation, md_file_path)
    """
    if not audio_file:
        return "Please upload an MP3 file.", None, None

    temp_wav = "./temp_input.wav"

    try:
        print("\n" + "=" * 60)
        print("Starting audio processing...")
        print("=" * 60)

        # Step 1: Convert MP3 to WAV
        convert_mp3_to_wav(audio_file, temp_wav)

        # Step 2: Transcribe
        chinese_text = transcribe_audio(temp_wav)

        # Step 3: Translate (using OpenAI wrapper)
        english_translation = translate_chinese_to_english_openai(chinese_text)

        # Step 4: Generate MD file
        md_file = generate_markdown_file(os.path.basename(audio_file), chinese_text, english_translation)

        print("=" * 60)
        print("✅ Processing completed successfully!")
        print("=" * 60 + "\n")

        return chinese_text, english_translation, md_file

    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, "", None

    finally:
        # Clean up temp file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            print("🗑️  Cleaned up temporary files")

# ========================================
# Launch Gradio App
# ========================================
with gr.Blocks(title="🎙️ Audio Translator (OpenAI Wrapper)") as demo:
    gr.Markdown("# 🎙️ Chinese Audio to English Translator")
    gr.Markdown(
        "Upload a **Chinese MP3 audio file**, and this tool will: "
        "**transcribe it to Chinese text**, then **translate to English** using OpenAI-compatible API, "
        "and export results as a **Markdown (.md) file**."
    )

    with gr.Row():
        audio_input = gr.File(label="Upload MP3 Audio", file_types=["audio"])

    btn = gr.Button("🚀 Transcribe & Translate")

    with gr.Row():
        ch_output = gr.Textbox(label="🇨🇳 Chinese Transcript", lines=6)
        en_output = gr.Textbox(label="🇬🇧 English Translation", lines=6)

    md_output = gr.File(label="📥 Download Markdown Report")

    btn.click(
        fn=process_audio_file,
        inputs=audio_input,
        outputs=[ch_output, en_output, md_output]
    )

    gr.Markdown("💡 Powered by **ModelScope (Paraformer)** + **Qwen API (OpenAI Wrapper)**")

# Run the app
if __name__ == "__main__":
    import torch  # Import here to check device
    print("\n" + "=" * 60)
    print("🚀 Starting Gradio App (app2.py - OpenAI Wrapper)")
    print("=" * 60)
    demo.launch(share=False)  # Set share=True for public link