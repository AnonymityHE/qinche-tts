import subprocess, os

TOKEN = "nTcN1cTgOeD5M0z5iKa_9KlXPTPPYeBpyaqJ4OJI"
ACCOUNT = "040b706a29a1319752dc78889ef4b3f5"
BUCKET = "tencent-tts"

uploads = [
    ("ft_v5_fast_checkpoint-epoch-3", "qwen3_v5_fast"),
    ("ft_v5_native_checkpoint-epoch-3", "qwen3_v5_native"),
    ("fish_s2_zeroshot", "fish_s2_zeroshot"),
    ("fish_s2_zeroshot_compile", "fish_s2_compile"),
]

BASE = "/home/ubuntu/yunlin/TTS/eval"

for src_dir, dest_prefix in uploads:
    for i in range(3):  # gen_00, gen_01, gen_02
        fname = f"gen_{i:02d}.wav"
        local = f"{BASE}/{src_dir}/{fname}"
        key = f"{dest_prefix}/{fname}"
        url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/r2/buckets/{BUCKET}/objects/{key}"
        cmd = [
            "curl", "-s", "-X", "PUT", url,
            "-H", f"Authorization: Bearer {TOKEN}",
            "-H", "Content-Type: audio/wav",
            "--data-binary", f"@{local}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        ok = '"success":true' in result.stdout or result.returncode == 0
        print(f"{'✓' if ok else '✗'} {key}  {result.stdout[:80]}")
