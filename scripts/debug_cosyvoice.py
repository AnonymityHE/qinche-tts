"""Debug CosyVoice LLM token generation to identify hallucination root cause."""
import sys
import json
import warnings
import uuid as _uuid
from collections import Counter

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice")
sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice/third_party/Matcha-TTS")

import torch
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

for i, part in enumerate(tts_text_parts):
    print(f"Part {i}: {part}")
    model_input = model.frontend.frontend_zero_shot(part, prompt_text_norm, ref_audio, sr, "")

    print(f"  Keys: {list(model_input.keys())}")
    for k, v in model_input.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")

    text_tensor = model_input["text"]
    print(f"  text tokens (first 30): {text_tensor[0][:30].tolist()}")
    pt_tensor = model_input["prompt_text"]
    print(f"  prompt_text tokens (first 30): {pt_tensor[0][:30].tolist()}")

    llm_spt = model_input["llm_prompt_speech_token"]
    print(f"  llm_prompt_speech_token: shape={llm_spt.shape}, first 20={llm_spt[0][:20].tolist()}")

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
        tokens = []
        for tok in token_gen:
            tokens.append(tok)
            if len(tokens) >= 500:
                print("  WARNING: hit 500 token limit!")
                break

    print(f"  LLM generated {len(tokens)} tokens")
    if tokens:
        print(f"  First 30 tokens: {tokens[:30]}")
        print(f"  Last 10 tokens: {tokens[-10:]}")
        print(f"  Unique token count: {len(set(tokens))}")
        c = Counter(tokens)
        print(f"  Most common: {c.most_common(5)}")

    # Also check what the speech_token_size is
    print(f"  speech_token_size: {inner.llm.speech_token_size}")
    print(f"  stop_token_ids: {inner.llm.stop_token_ids}")
    print(f"  fp16: {inner.fp16}")
    break
