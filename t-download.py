# test_asr.py
from modelscope.pipelines import pipeline
import torch

print("CUDA available:", torch.cuda.is_available())
device = 'gpu' if torch.cuda.is_available() else 'cpu'

pipe = pipeline(
    task='auto-speech-recognition',
    model='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    device=device
)

# Replace with your actual WAV file path
result = pipe("t-ngamuk.wav")
print("Transcription:", result["text"])