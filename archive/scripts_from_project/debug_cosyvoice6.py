"""Compare ref mel and synth mel distributions in detail."""
import sys
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice")
sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice/third_party/Matcha-TTS")

import torch
import numpy as np
from cosyvoice.cli.cosyvoice import AutoModel

model = AutoModel(model_dir="/home/ubuntu/yunlin/TTS/models/cosyvoice/Fun-CosyVoice3-0.5B-2512")
sr = model.sample_rate
inner = model.model

refs = json.load(open("/home/ubuntu/yunlin/TTS/data/dataset/ref_manifest.json"))
ref_audio = refs[0]["audio_filepath"]
ref_text = refs[0]["text"]

# Get ref mel
ref_feat, _ = model.frontend._extract_speech_feat(ref_audio)
ref_mel = ref_feat.to(inner.device).transpose(1, 2)  # (1, 80, T)
print(f"Ref mel: shape={ref_mel.shape}")

# Get synth mel
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

    print(f"Synth mel: shape={synth_mel.shape}")

    ref_np = ref_mel.squeeze().cpu().float().numpy()
    synth_np = synth_mel.squeeze().cpu().float().numpy()

    print("\n=== Per-band comparison (80 mel bands) ===")
    print(f"{'Band':>5} {'Ref Mean':>10} {'Ref Std':>10} {'Syn Mean':>10} {'Syn Std':>10} {'Diff Mean':>10}")
    for b in range(0, 80, 10):
        rm = ref_np[b].mean()
        rs = ref_np[b].std()
        sm = synth_np[b].mean()
        ss = synth_np[b].std()
        print(f"{b:5d} {rm:10.3f} {rs:10.3f} {sm:10.3f} {ss:10.3f} {sm-rm:10.3f}")

    print("\n=== Overall stats ===")
    print(f"Ref:   mean={ref_np.mean():.3f}, std={ref_np.std():.3f}, min={ref_np.min():.3f}, max={ref_np.max():.3f}")
    print(f"Synth: mean={synth_np.mean():.3f}, std={synth_np.std():.3f}, min={synth_np.min():.3f}, max={synth_np.max():.3f}")

    # Check temporal structure: energy over time
    ref_energy = ref_np.mean(axis=0)
    synth_energy = synth_np.mean(axis=0)
    print(f"\nRef temporal energy: first 20 = {ref_energy[:20]}")
    print(f"Synth temporal energy: first 20 = {synth_energy[:20]}")

    # Check low-frequency bands (which carry f0 info)
    ref_low = ref_np[:10].mean(axis=0)
    synth_low = synth_np[:10].mean(axis=0)
    print(f"\nRef low-freq energy (mean): {ref_low.mean():.3f}, std: {ref_low.std():.3f}")
    print(f"Synth low-freq energy (mean): {synth_low.mean():.3f}, std: {synth_low.std():.3f}")

    # Check if synth mel has the same range as training distribution
    print(f"\nRef mel percentiles:  5%={np.percentile(ref_np, 5):.3f}, 25%={np.percentile(ref_np, 25):.3f}, 50%={np.percentile(ref_np, 50):.3f}, 75%={np.percentile(ref_np, 75):.3f}, 95%={np.percentile(ref_np, 95):.3f}")
    print(f"Synth mel percentiles: 5%={np.percentile(synth_np, 5):.3f}, 25%={np.percentile(synth_np, 25):.3f}, 50%={np.percentile(synth_np, 50):.3f}, 75%={np.percentile(synth_np, 75):.3f}, 95%={np.percentile(synth_np, 95):.3f}")

    break
