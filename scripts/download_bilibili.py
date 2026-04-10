"""Download audio from Bilibili videos using bilibili-api-python.

Supports multi-page videos. Each page is downloaded as a separate WAV.
Uses streaming download with retry for reliability.
"""
import asyncio
import os
import subprocess
import traceback

from bilibili_api import video, HEADERS
import httpx

SAVE_DIR = "/home/ubuntu/yunlin/TTS/data/raw"
MAX_RETRIES = 3

VIDEOS = [
    {"bvid": "BV1NjzAB2EPY", "name": "qinche_01"},
    {"bvid": "BV168UQBeEKp", "name": "qinche_02"},
    {"bvid": "BV1Z7HTecEBD", "name": "qinche_pure", "pages": list(range(11))},
]


async def download_with_retry(url: str, headers: dict, dest_path: str, name: str):
    """Stream-download a URL to a file with retries."""
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
            print(f"  [{name}] Download attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if attempt < MAX_RETRIES:
                wait = 5 * attempt
                print(f"  [{name}] Retrying in {wait}s...")
                await asyncio.sleep(wait)
    return False


async def download_page_audio(bvid: str, name: str, page_index: int = 0):
    """Download audio for a single page of a Bilibili video."""
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
        print(f"[{name}] FAILED after {MAX_RETRIES} attempts, skipping.")
        return
    print(f"[{name}] Downloaded {os.path.getsize(m4a_path) / 1024 / 1024:.1f} MB")

    print(f"[{name}] Converting to WAV (24kHz mono)...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", m4a_path, "-ar", "24000", "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )
    os.remove(m4a_path)
    print(f"[{name}] Done: {wav_path} ({os.path.getsize(wav_path) / 1024 / 1024:.1f} MB)")


async def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    for v in VIDEOS:
        pages = v.get("pages")
        if pages is not None:
            for p in pages:
                page_name = f"{v['name']}_p{p+1:02d}"
                try:
                    await download_page_audio(v["bvid"], page_name, page_index=p)
                except Exception as e:
                    print(f"[{page_name}] Error: {e}")
                    traceback.print_exc()
        else:
            try:
                await download_page_audio(v["bvid"], v["name"])
            except Exception as e:
                print(f"[{v['name']}] Error: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
