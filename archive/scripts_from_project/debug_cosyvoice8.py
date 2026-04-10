"""Compare LLM speech tokens from synth vs extracted from ref audio."""
import sys
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice")
sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice/third_party/Matcha-TTS")

import torch
import numpy as np
from collections import Counter
from cosyvoice.cli.cosyvoice import AutoModel

model = AutoModel(model_dir="/home/ubuntu/yunlin/TTS/models/cosyvoice/Fun-CosyVoice3-0.5B-2512")
sr = model.sample_rate
inner = model.model

refs = json.load(open("/home/ubuntu/yunlin/TTS/data/dataset/ref_manifest.json"))
ref_audio = refs[0]["audio_filepath"]
ref_text = refs[0]["text"]

# Get speech tokens extracted from ref audio (ground truth)
ref_speech_token, ref_speech_token_len = model.frontend._extract_speech_token(ref_audio)
print(f"Ref speech token: shape={ref_speech_token.shape}, len={ref_speech_token_len}")
ref_tokens = ref_speech_token.squeeze().tolist()
print(f"Ref tokens (first 30): {ref_tokens[:30]}")
print(f"Ref token range: min={min(ref_tokens)}, max={max(ref_tokens)}")
ref_counter = Counter(ref_tokens)
print(f"Ref unique tokens: {len(ref_counter)}")
print(f"Ref most common: {ref_counter.most_common(5)}")

# Get LLM generated speech tokens
text = "你好，今天天气真不错。"
prompt_text = "You are a helpful assistant.<|endofprompt|>" + ref_text

prompt_text_norm = model.frontend.text_normalize(prompt_text, split=False, text_frontend=False)
tts_text_parts = list(model.frontend.text_normalize(text, split=True, text_frontend=False))

for part in tts_text_parts:
    model_input = model.frontend.frontend_zero_shot(part, prompt_text_norm, ref_audio, sr, "")

    # Check prompt speech tokens (extracted from ref, used as LLM prompt)
    prompt_tokens = model_input["llm_prompt_speech_token"].squeeze().tolist()
    print(f"\nPrompt speech tokens (same as ref): {prompt_tokens[:30]}")
    print(f"Match ref? {prompt_tokens[:10] == ref_tokens[:10]}")

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

    print(f"\nLLM generated tokens (first 30): {all_tokens[:30]}")
    print(f"LLM token range: min={min(all_tokens)}, max={max(all_tokens)}")
    gen_counter = Counter(all_tokens)
    print(f"LLM unique tokens: {len(gen_counter)}")
    print(f"LLM most common: {gen_counter.most_common(5)}")
    
    # Distribution comparison
    print(f"\nRef token distribution: mean={np.mean(ref_tokens):.1f}, std={np.std(ref_tokens):.1f}")
    print(f"Gen token distribution: mean={np.mean(all_tokens):.1f}, std={np.std(all_tokens):.1f}")

    # Check speech_token_size
    print(f"\nspeech_token_size: {inner.llm.speech_token_size}")
    print(f"Any generated token >= speech_token_size? {any(t >= inner.llm.speech_token_size for t in all_tokens)}")

    break
