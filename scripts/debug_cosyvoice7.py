"""Test if mel shift fixes f0 and final audio quality."""
import sys
import json
import warnings

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
        synth_mel, _ = inner.flow.inference(
            token=token_tensor.to(inner.device, dtype=torch.int32),
            token_len=torch.tensor([token_tensor.shape[1]], dtype=torch.int32).to(inner.device),
            prompt_token=model_input["flow_prompt_speech_token"].to(inner.device),
            prompt_token_len=torch.tensor([model_input["flow_prompt_speech_token"].shape[1]], dtype=torch.int32).to(inner.device),
            prompt_feat=model_input["prompt_speech_feat"].to(inner.device),
            prompt_feat_len=torch.tensor([model_input["prompt_speech_feat"].shape[1]], dtype=torch.int32).to(inner.device),
            embedding=model_input["flow_embedding"].to(inner.device),
            streaming=False, finalize=True,
        )

    print(f"Synth mel: {synth_mel.shape}, mean={synth_mel.mean():.3f}")

    # Test 1: original synth mel
    with torch.inference_mode():
        hift.f0_predictor.to(torch.float64)
        speech1, _ = hift.inference(speech_feat=synth_mel, finalize=True)
        audio1 = speech1.squeeze().cpu().float().numpy()
        sf.write("/tmp/cv3_orig.wav", audio1, sr)
        
        f0_1 = hift.f0_predictor(synth_mel.to(torch.float64), finalize=True)
        voiced1 = (f0_1 > 50).float().mean()
        print(f"Original: voiced={voiced1:.3f}, audio_std={audio1.std():.4f}")

    # Test 2: shift mel +1
    shifted_mel = synth_mel + 1.0
    with torch.inference_mode():
        speech2, _ = hift.inference(speech_feat=shifted_mel, finalize=True)
        audio2 = speech2.squeeze().cpu().float().numpy()
        sf.write("/tmp/cv3_shift1.wav", audio2, sr)
        
        f0_2 = hift.f0_predictor(shifted_mel.to(torch.float64), finalize=True)
        voiced2 = (f0_2 > 50).float().mean()
        print(f"Shift +1: voiced={voiced2:.3f}, audio_std={audio2.std():.4f}")

    # Test 3: shift mel +2
    shifted_mel3 = synth_mel + 2.0
    with torch.inference_mode():
        speech3, _ = hift.inference(speech_feat=shifted_mel3, finalize=True)
        audio3 = speech3.squeeze().cpu().float().numpy()
        sf.write("/tmp/cv3_shift2.wav", audio3, sr)
        
        f0_3 = hift.f0_predictor(shifted_mel3.to(torch.float64), finalize=True)
        voiced3 = (f0_3 > 50).float().mean()
        print(f"Shift +2: voiced={voiced3:.3f}, audio_std={audio3.std():.4f}")

    # ASR check all three
    import whisperx
    asr = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="zh")
    
    for name, path in [("orig", "/tmp/cv3_orig.wav"), ("+1", "/tmp/cv3_shift1.wav"), ("+2", "/tmp/cv3_shift2.wav")]:
        wav = whisperx.load_audio(path)
        result = asr.transcribe(wav, batch_size=4, language="zh")
        hyp = "".join(seg["text"] for seg in result["segments"]).strip()
        print(f"  {name}: {hyp}")

    break
