"""Debug CosyVoice f0 predictor and source signal in vocoder."""
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
hift = inner.hift

refs = json.load(open("/home/ubuntu/yunlin/TTS/data/dataset/ref_manifest.json"))
ref_audio = refs[0]["audio_filepath"]
ref_text = refs[0]["text"]

text = "你好，今天天气真不错。"
prompt_text = "You are a helpful assistant.<|endofprompt|>" + ref_text

# Run normal inference to get mel
prompt_text_norm = model.frontend.text_normalize(prompt_text, split=False, text_frontend=False)
tts_text_parts = list(model.frontend.text_normalize(text, split=True, text_frontend=False))

for part in tts_text_parts:
    model_input = model.frontend.frontend_zero_shot(part, prompt_text_norm, ref_audio, sr, "")

    # Get LLM tokens
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
        all_tokens = list(token_gen)

    token_tensor = torch.tensor([all_tokens], dtype=torch.int32)

    # Get mel from flow
    with torch.cuda.amp.autocast(inner.fp16):
        tts_mel, _ = inner.flow.inference(
            token=token_tensor.to(inner.device, dtype=torch.int32),
            token_len=torch.tensor([token_tensor.shape[1]], dtype=torch.int32).to(inner.device),
            prompt_token=model_input["flow_prompt_speech_token"].to(inner.device),
            prompt_token_len=torch.tensor([model_input["flow_prompt_speech_token"].shape[1]], dtype=torch.int32).to(inner.device),
            prompt_feat=model_input["prompt_speech_feat"].to(inner.device),
            prompt_feat_len=torch.tensor([model_input["prompt_speech_feat"].shape[1]], dtype=torch.int32).to(inner.device),
            embedding=model_input["flow_embedding"].to(inner.device),
            streaming=False,
            finalize=True,
        )

    print(f"Mel: shape={tts_mel.shape}, dtype={tts_mel.dtype}")

    # Now manually step through vocoder
    speech_feat = tts_mel

    # Step 1: f0 prediction
    print(f"\nf0_predictor type: {type(hift.f0_predictor).__name__}")
    print(f"f0_predictor device: {next(hift.f0_predictor.parameters()).device}")
    print(f"f0_predictor dtype: {next(hift.f0_predictor.parameters()).dtype}")

    # Test f0 with float32 first
    with torch.inference_mode():
        f0_f32 = hift.f0_predictor(speech_feat)
        print(f"\nf0 (float32): shape={f0_f32.shape}, min={f0_f32.min():.2f}, max={f0_f32.max():.2f}, mean={f0_f32.mean():.2f}")
        print(f"f0 (float32) first 20 values: {f0_f32[0, :20].cpu().numpy()}")
        
        # Test with float64 (as in the latest code)
        hift.f0_predictor.to(torch.float64)
        f0_f64 = hift.f0_predictor(speech_feat.to(torch.float64), finalize=True).to(speech_feat)
        print(f"\nf0 (float64): shape={f0_f64.shape}, min={f0_f64.min():.2f}, max={f0_f64.max():.2f}, mean={f0_f64.mean():.2f}")
        print(f"f0 (float64) first 20 values: {f0_f64[0, :20].cpu().numpy()}")
        
        # Difference
        diff = (f0_f32 - f0_f64).abs()
        print(f"\nf0 diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

    # Step 2: source signal
    with torch.inference_mode():
        hift.f0_predictor.to(torch.float64)
        f0 = hift.f0_predictor(speech_feat.to(torch.float64), finalize=True).to(speech_feat)
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        print(f"\nUpsampled f0: shape={s.shape}, min={s.min():.2f}, max={s.max():.2f}")
        
        s_out, _, _ = hift.m_source(s)
        s_out = s_out.transpose(1, 2)
        print(f"Source signal: shape={s_out.shape}, min={s_out.min():.4f}, max={s_out.max():.4f}, std={s_out.std():.6f}")

    # Step 3: Full vocoder output
    with torch.inference_mode():
        hift.f0_predictor.to(torch.float64)
        gen_speech, gen_source = hift.inference(speech_feat=speech_feat, finalize=True)
        audio_np = gen_speech.squeeze().cpu().float().numpy()
        print(f"\nFinal audio: {len(audio_np)/sr:.2f}s, std={audio_np.std():.4f}")
        sf.write("/tmp/cv3_debug_v4.wav", audio_np, sr)

    # Also test: what if we use the reference audio's mel through the same vocoder?
    ref_wav_data, ref_sr = sf.read(ref_audio)
    print(f"\nRef audio: {len(ref_wav_data)/ref_sr:.2f}s")

    break
