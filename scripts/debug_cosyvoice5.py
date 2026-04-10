"""Test f0 predictor with reference audio's mel to see if it's the predictor or the mel."""
import sys
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice")
sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice/third_party/Matcha-TTS")

import torch
import numpy as np
import soundfile as sf
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

model = AutoModel(model_dir="/home/ubuntu/yunlin/TTS/models/cosyvoice/Fun-CosyVoice3-0.5B-2512")
sr = model.sample_rate
inner = model.model
hift = inner.hift

refs = json.load(open("/home/ubuntu/yunlin/TTS/data/dataset/ref_manifest.json"))
ref_audio = refs[0]["audio_filepath"]

# Load ref audio and extract mel using the same frontend
ref_wav, ref_sr = torchaudio.load(ref_audio, backend='soundfile')
if ref_sr != sr:
    ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, sr)
if ref_wav.shape[0] > 1:
    ref_wav = ref_wav.mean(dim=0, keepdim=True)

print(f"Ref wav: shape={ref_wav.shape}, sr={sr}")

# Extract mel features using the frontend
ref_feat, ref_feat_len = model.frontend._extract_speech_feat(ref_audio)
print(f"Ref mel (from frontend): shape={ref_feat.shape}")

# Run f0 predictor on ref mel
with torch.inference_mode():
    ref_mel_gpu = ref_feat.to(inner.device).transpose(1, 2)  # (B, 80, T)
    print(f"Ref mel for vocoder: shape={ref_mel_gpu.shape}")
    
    hift.f0_predictor.to(torch.float64)
    f0_ref = hift.f0_predictor(ref_mel_gpu.to(torch.float64), finalize=True).to(ref_mel_gpu)
    print(f"Ref f0: shape={f0_ref.shape}, min={f0_ref.min():.2f}, max={f0_ref.max():.2f}, mean={f0_ref.mean():.2f}")
    print(f"Ref f0 first 20: {f0_ref[0, :20].cpu().numpy()}")
    
    voiced = (f0_ref > 50).float()
    print(f"Ref voiced fraction: {voiced.mean():.3f}")

# Now also run f0 on the synthesized mel to compare
text = "你好，今天天气真不错。"
ref_text = refs[0]["text"]
prompt_text = "You are a helpful assistant.<|endofprompt|>" + ref_text

prompt_text_norm = model.frontend.text_normalize(prompt_text, split=False, text_frontend=False)
tts_text_parts = list(model.frontend.text_normalize(text, split=True, text_frontend=False))

for part in tts_text_parts:
    model_input = model.frontend.frontend_zero_shot(part, prompt_text_norm, ref_audio, sr, "")

    with torch.cuda.amp.autocast(inner.fp16 is True and not hasattr(inner.llm, "vllm")):
        all_tokens = list(inner.llm.inference(
            text=model_input["text"].to(inner.device),
            text_len=torch.tensor([model_input["text"].shape[1]], dtype=torch.int32).to(inner.device),
            prompt_text=model_input["prompt_text"].to(inner.device),
            prompt_text_len=torch.tensor([model_input["prompt_text"].shape[1]], dtype=torch.int32).to(inner.device),
            prompt_speech_token=model_input["llm_prompt_speech_token"].to(inner.device),
            prompt_speech_token_len=torch.tensor([model_input["llm_prompt_speech_token"].shape[1]], dtype=torch.int32).to(inner.device),
            embedding=model_input["llm_embedding"].to(inner.device),
        ))

    token_tensor = torch.tensor([all_tokens], dtype=torch.int32)

    with torch.cuda.amp.autocast(inner.fp16):
        tts_mel, _ = inner.flow.inference(
            token=token_tensor.to(inner.device, dtype=torch.int32),
            token_len=torch.tensor([token_tensor.shape[1]], dtype=torch.int32).to(inner.device),
            prompt_token=model_input["flow_prompt_speech_token"].to(inner.device),
            prompt_token_len=torch.tensor([model_input["flow_prompt_speech_token"].shape[1]], dtype=torch.int32).to(inner.device),
            prompt_feat=model_input["prompt_speech_feat"].to(inner.device),
            prompt_feat_len=torch.tensor([model_input["prompt_speech_feat"].shape[1]], dtype=torch.int32).to(inner.device),
            embedding=model_input["flow_embedding"].to(inner.device),
            streaming=False, finalize=True,
        )

    with torch.inference_mode():
        hift.f0_predictor.to(torch.float64)
        f0_synth = hift.f0_predictor(tts_mel.to(torch.float64), finalize=True).to(tts_mel)
        print(f"\nSynth f0: shape={f0_synth.shape}, min={f0_synth.min():.2f}, max={f0_synth.max():.2f}, mean={f0_synth.mean():.2f}")
        print(f"Synth f0 first 20: {f0_synth[0, :20].cpu().numpy()}")
        
        voiced_synth = (f0_synth > 50).float()
        print(f"Synth voiced fraction: {voiced_synth.mean():.3f}")

    # Reconstruct ref audio through vocoder as sanity check
    with torch.inference_mode():
        ref_mel_for_voc = ref_mel_gpu.float()
        recon_speech, _ = hift.inference(speech_feat=ref_mel_for_voc, finalize=True)
        recon_np = recon_speech.squeeze().cpu().float().numpy()
        sf.write("/tmp/cv3_recon_ref.wav", recon_np, sr)
        print(f"\nRecon ref audio: {len(recon_np)/sr:.2f}s, std={recon_np.std():.4f}")

    # ASR on reconstructed ref
    import whisperx
    asr = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="zh")
    wav = whisperx.load_audio("/tmp/cv3_recon_ref.wav")
    result = asr.transcribe(wav, batch_size=4, language="zh")
    hyp = "".join(seg["text"] for seg in result["segments"]).strip()
    print(f"Ref text: {ref_text}")
    print(f"Recon HYP: {hyp}")

    break
