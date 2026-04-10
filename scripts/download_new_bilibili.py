"""Download audio from the new Bilibili video (语音通话合集, 6 pages)."""
import asyncio
import os
import subprocess
import traceback

from bilibili_api import video, HEADERS
import httpx

SAVE_DIR = "/home/ubuntu/yunlin/TTS/data/raw"
MAX_RETRIES = 3


async def download_with_retry(url: str, headers: dict, dest_path: str, name: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(
                headers=headers, follow_redirects=True, timeout=httpx.Timeout(30, read=120)
            ) as client:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(dest_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
            return True
        except Exception as e:
            print(f"  [{name}] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(5 * attempt)
    return False


async def download_page_audio(bvid: str, name: str, page_index: int = 0):
    wav_path = os.path.join(SAVE_DIR, f"{name}.wav")
    if os.path.exists(wav_path):
        print(f"[{name}] Already exists, skipping.")
        return

    v = video.Video(bvid=bvid)
    info = await v.get_info()
    pages = info.get("pages", [])
    page_title = pages[page_index]["part"] if page_index < len(pages) else "?"
    page_dur = pages[page_index]["duration"] if page_index < len(pages) else "?"
    print(f"[{name}] Page {page_index}: {page_title} ({page_dur}s)")

    download_url_data = await v.get_download_url(page_index)
    audio_streams = download_url_data.get("dash", {}).get("audio", [])
    if not audio_streams:
        print(f"[{name}] No audio streams found!")
        return

    audio_streams.sort(key=lambda x: x.get("bandwidth", 0), reverse=True)
    audio_url = audio_streams[0]["baseUrl"]

    headers = dict(HEADERS)
    headers["Referer"] = f"https://www.bilibili.com/video/{bvid}/"

    m4a_path = os.path.join(SAVE_DIR, f"{name}.m4a")

    print(f"[{name}] Downloading audio...")
    ok = await download_with_retry(audio_url, headers, m4a_path, name)
    if not ok:
        print(f"[{name}] FAILED, skipping.")
        return
    print(f"[{name}] Downloaded {os.path.getsize(m4a_path) / 1024 / 1024:.1f} MB")

    print(f"[{name}] Converting to WAV (24kHz mono)...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", m4a_path, "-ar", "24000", "-ac", "1", wav_path],
        check=True, capture_output=True,
    )
    os.remove(m4a_path)
    print(f"[{name}] Done: {wav_path} ({os.path.getsize(wav_path) / 1024 / 1024:.1f} MB)")


async def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    bvid = "BV1JNCBB1Epq"
    for p in range(6):
        name = f"qinche_call_p{p+1:02d}"
        try:
            await download_page_audio(bvid, name, page_index=p)
        except Exception as e:
            print(f"[{name}] Error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
