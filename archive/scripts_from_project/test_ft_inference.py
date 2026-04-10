"""Quick test: can finetuned checkpoint generate audio at different text lengths?"""
import torch, soundfile as sf, time
from qwen_tts import Qwen3TTSModel

CKPT = "/home/ubuntu/yunlin/TTS/output/qinche_sft/checkpoint-epoch-4"

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    CKPT, device_map="cuda:0", dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
print(f"Speakers: {model.get_supported_speakers()}")
print(f"Languages: {model.get_supported_languages()}")

tests = [
    "你好。",
    "今天天气不错。",
    "放心吧，这件事情交给我来处理，你只需要好好休息就行了。",
]

for i, text in enumerate(tests):
    print(f"\n[{i}] Generating: {text} (len={len(text)})")
    t0 = time.time()
    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker="qinche",
            language="Chinese",
            max_new_tokens=2048,
        )
        elapsed = time.time() - t0
        dur = len(wavs[0]) / sr
        print(f"    OK: {dur:.2f}s audio in {elapsed:.1f}s (RTF={elapsed/dur:.2f})")
        sf.write(f"/tmp/test_ft_{i}.wav", wavs[0], sr)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    ERROR after {elapsed:.1f}s: {e}")

print("\nDONE")
