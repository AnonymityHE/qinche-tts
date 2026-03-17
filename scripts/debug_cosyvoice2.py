"""Debug CosyVoice token2wav pipeline."""
import sys
import json
import warnings
import uuid as _uuid

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice")
sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice/third_party/Matcha-TTS")

import torch
import numpy as np
import soundfile as sf
from cosyvoice.cli.cosyvoice import AutoModel

model = AutoModel(model_dir="/home/ubuntu/yunlin/TTS/models/cosyvoice/Fun-CosyVoice3-0.5B-2512")
sr = model.sample_rate
inner = model.model

refs = json.load(open("/home/ubuntu/yunlin/TTS/data/dataset/ref_manifest.json"))
ref_audio = refs[0]["audio_filepath"]
ref_text = refs[0]["text"]

text = "你好，今天天气真不错。"
prompt_text = "You are a helpful assistant.<|endofprompt|>" + ref_text

# Normal inference path - collect the audio
chunks = []
for c in model.inference_zero_shot(text, prompt_text, ref_audio, stream=False, text_frontend=False):
    chunks.append(c["tts_speech"])

if chunks:
    audio = torch.cat(chunks, dim=-1).squeeze().cpu().float().numpy()
    sf.write("/tmp/cv3_debug.wav", audio, sr)
    print(f"Audio: {len(audio)/sr:.2f}s, min={audio.min():.4f}, max={audio.max():.4f}, std={audio.std():.4f}")
    
    # Check if audio is very low energy or clipped
    rms = np.sqrt(np.mean(audio**2))
    print(f"RMS: {rms:.6f}")
    
    # Check spectral content
    from scipy import signal
    freqs, times, Sxx = signal.spectrogram(audio, fs=sr, nperseg=1024)
    mean_power = np.mean(Sxx, axis=1)
    peak_freq = freqs[np.argmax(mean_power)]
    print(f"Peak frequency: {peak_freq:.1f} Hz")
    print(f"Mean spectral power shape: {mean_power.shape}")
    
    # Check for repetitive patterns in audio
    # Compute autocorrelation on a chunk
    chunk_size = sr  # 1 second
    chunk = audio[:chunk_size]
    autocorr = np.correlate(chunk, chunk, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    
    # Find peaks (repetitive patterns)
    peaks = signal.find_peaks(autocorr, height=0.5, distance=100)[0]
    if len(peaks) > 0:
        print(f"Autocorrelation peaks at samples: {peaks[:10]}")
        periods = [f"{sr/p:.1f}" for p in peaks[:5]]
        print(f"Corresponding frequencies: {periods} Hz")
    else:
        print("No strong autocorrelation peaks found")
else:
    print("No audio generated!")

# Now test with CosyVoice2
print("\n--- CosyVoice2 ---")
model2 = AutoModel(model_dir="/home/ubuntu/yunlin/TTS/models/cosyvoice/CosyVoice2-0.5B")
sr2 = model2.sample_rate

chunks2 = []
for c in model2.inference_zero_shot(text, ref_text, ref_audio, stream=False, text_frontend=False):
    chunks2.append(c["tts_speech"])

if chunks2:
    audio2 = torch.cat(chunks2, dim=-1).squeeze().cpu().float().numpy()
    sf.write("/tmp/cv2_debug.wav", audio2, sr2)
    print(f"Audio: {len(audio2)/sr2:.2f}s, min={audio2.min():.4f}, max={audio2.max():.4f}, std={audio2.std():.4f}")
    rms2 = np.sqrt(np.mean(audio2**2))
    print(f"RMS: {rms2:.6f}")
    
    import whisperx
    asr = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="zh")
    
    wav3 = whisperx.load_audio("/tmp/cv3_debug.wav")
    result3 = asr.transcribe(wav3, batch_size=4, language="zh")
    hyp3 = "".join(seg["text"] for seg in result3["segments"]).strip()
    
    wav2 = whisperx.load_audio("/tmp/cv2_debug.wav")
    result2 = asr.transcribe(wav2, batch_size=4, language="zh")
    hyp2 = "".join(seg["text"] for seg in result2["segments"]).strip()
    
    print(f"\nGT:  {text}")
    print(f"CV3: {hyp3}")
    print(f"CV2: {hyp2}")
