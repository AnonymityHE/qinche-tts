"""Deep debug: check mel spectrograms from flow matching."""
import sys
import json
import warnings
import threading
import uuid as _uuid
from contextlib import nullcontext

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

prompt_text_norm = model.frontend.text_normalize(prompt_text, split=False, text_frontend=False)
tts_text_parts = list(model.frontend.text_normalize(text, split=True, text_frontend=False))

for part in tts_text_parts:
    model_input = model.frontend.frontend_zero_shot(part, prompt_text_norm, ref_audio, sr, "")

    # Manually run LLM to get tokens
    uid = str(_uuid.uuid1())
    inner.tts_speech_token_dict[uid] = []
    inner.llm_end_dict[uid] = False
    inner.hift_cache_dict[uid] = None

    with torch.cuda.amp.autocast(inner.fp16 is True and not hasattr(inner.llm, "vllm")):
        token_gen = inner.llm.inference(
            text=model_input["text"].to(inner.device),
            text_len=torch.tensor([model_input["text"].shape[1]], dtype=torch.int32).to(inner.device),
            prompt_text=model_input["prompt_text"].to(inner.device),
            prompt_text_len=torch.tensor([model_input["prompt_text"].shape[1]], dtype=torch.int32).to(inner.device),
            prompt_speech_token=model_input["llm_prompt_speech_token"].to(inner.device),
            prompt_speech_token_len=torch.tensor([model_input["llm_prompt_speech_token"].shape[1]], dtype=torch.int32).to(inner.device),
            embedding=model_input["llm_embedding"].to(inner.device),
        )
        all_tokens = []
        for tok in token_gen:
            all_tokens.append(tok)

    print(f"LLM generated {len(all_tokens)} tokens")
    token_tensor = torch.tensor([all_tokens], dtype=torch.int32)
    print(f"Token tensor shape: {token_tensor.shape}")

    # Run flow matching (token2wav) manually
    prompt_token = model_input["flow_prompt_speech_token"]
    prompt_feat = model_input["prompt_speech_feat"]
    embedding = model_input["flow_embedding"]

    print(f"prompt_token shape: {prompt_token.shape}")
    print(f"prompt_feat shape: {prompt_feat.shape}")
    print(f"embedding shape: {embedding.shape}")

    with torch.cuda.amp.autocast(inner.fp16):
        tts_mel, _ = inner.flow.inference(
            token=token_tensor.to(inner.device, dtype=torch.int32),
            token_len=torch.tensor([token_tensor.shape[1]], dtype=torch.int32).to(inner.device),
            prompt_token=prompt_token.to(inner.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(inner.device),
            prompt_feat=prompt_feat.to(inner.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(inner.device),
            embedding=embedding.to(inner.device),
            streaming=False,
            finalize=True,
        )

    print(f"Mel shape: {tts_mel.shape}, dtype: {tts_mel.dtype}")
    mel_np = tts_mel.squeeze().cpu().float().numpy()
    print(f"Mel stats: min={mel_np.min():.4f}, max={mel_np.max():.4f}, mean={mel_np.mean():.4f}, std={mel_np.std():.4f}")

    # Check if mel has variation across time
    mel_var_time = np.var(mel_np, axis=0)  # variance across frequency for each time step
    print(f"Mel variance across time: min={mel_var_time.min():.6f}, max={mel_var_time.max():.6f}, mean={mel_var_time.mean():.6f}")

    mel_var_freq = np.var(mel_np, axis=1)  # variance across time for each freq bin
    print(f"Mel variance across freq bins (first 10): {mel_var_freq[:10]}")

    # Check for repetitive mel frames
    if mel_np.shape[1] > 10:
        frame_diffs = np.diff(mel_np, axis=1)
        avg_frame_diff = np.mean(np.abs(frame_diffs))
        print(f"Average frame-to-frame difference: {avg_frame_diff:.6f}")

    # Run HiFT vocoder (CosyVoice3 uses CausalHiFTGenerator)
    tts_speech, tts_source = inner.hift.inference(speech_feat=tts_mel, finalize=True)
    audio_np = tts_speech.squeeze().cpu().float().numpy()
    sf.write("/tmp/cv3_debug_manual.wav", audio_np, sr)
    print(f"\nVocoder output: {len(audio_np)/sr:.2f}s, min={audio_np.min():.4f}, max={audio_np.max():.4f}, std={audio_np.std():.4f}")
    rms = np.sqrt(np.mean(audio_np**2))
    print(f"RMS: {rms:.6f}")

    # Quick ASR check on the manually generated audio
    import whisperx
    asr = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="zh")
    wav = whisperx.load_audio("/tmp/cv3_debug_manual.wav")
    result = asr.transcribe(wav, batch_size=4, language="zh")
    hyp = "".join(seg["text"] for seg in result["segments"]).strip()
    print(f"GT:  {text}")
    print(f"HYP: {hyp}")
    
    break
