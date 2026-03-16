#!/usr/bin/env python3
"""Upload dist/ to Cloudflare Pages via Direct Upload API (form-based)."""
import os, sys, hashlib, mimetypes, json, ssl
import urllib.request

TOKEN   = "nTcN1cTgOeD5M0z5iKa_9KlXPTPPYeBpyaqJ4OJI"
ACCOUNT = "040b706a29a1319752dc78889ef4b3f5"
PROJECT = "qinche-tts"
DIST    = os.path.join(os.path.dirname(__file__), "dist")
CTX     = ssl.create_default_context()

def cf_request(method, path, body=None, content_type=None):
    url = f"https://api.cloudflare.com/client/v4{path}"
    hdrs = {"Authorization": f"Bearer {TOKEN}"}
    if content_type:
        hdrs["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, context=CTX) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())

# 1. Collect all files
files = {}
for root, _, fnames in os.walk(DIST):
    for fname in fnames:
        fpath = os.path.join(root, fname)
        rel = "/" + os.path.relpath(fpath, DIST).replace(os.sep, "/")
        with open(fpath, "rb") as f:
            content = f.read()
        h = hashlib.sha256(content).hexdigest()
        files[rel] = {"path": fpath, "hash": h, "content": content,
                      "mime": mimetypes.guess_type(rel)[0] or "application/octet-stream"}

print(f"Files: {len(files)}")

# 2. Create deployment via multipart (Pages Direct Upload)
BOUNDARY = "----CFPagesBoundary01"

def make_multipart(fields):
    parts = b""
    for name, value in fields:
        parts += f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode()
        parts += (value if isinstance(value, bytes) else value.encode())
        parts += b"\r\n"
    parts += f"--{BOUNDARY}--\r\n".encode()
    return parts

# Build manifest: {"/index.html": "<hash>", ...}
manifest = {rel: info["hash"] for rel, info in files.items()}

body = make_multipart([("manifest", json.dumps(manifest))])
resp = cf_request("POST",
    f"/accounts/{ACCOUNT}/pages/projects/{PROJECT}/deployments",
    body=body,
    content_type=f"multipart/form-data; boundary={BOUNDARY}")

print("Create deployment:", resp.get("success"), resp.get("errors"))
if not resp.get("success"):
    print(json.dumps(resp, indent=2)); sys.exit(1)

dep_id = resp["result"]["id"]
print(f"Deployment ID: {dep_id}")

# 3. Upload each file
print("Uploading files...")
for rel, info in files.items():
    FBOUNDARY = "----CFFileBoundary" + info["hash"][:8]
    fbody = (
        f"--{FBOUNDARY}\r\n"
        f"Content-Disposition: form-data; name=\"{info['hash']}\"; filename=\"{info['hash']}\"\r\n"
        f"Content-Type: {info['mime']}\r\n\r\n"
    ).encode() + info["content"] + f"\r\n--{FBOUNDARY}--\r\n".encode()

    r = cf_request("POST",
        f"/accounts/{ACCOUNT}/pages/projects/{PROJECT}/deployments/{dep_id}/files",
        body=fbody,
        content_type=f"multipart/form-data; boundary={FBOUNDARY}")
    print(f"  {'✓' if r.get('success') else '✗'} {rel}")

# 4. Finalize
print("Finalizing deployment...")
fin = cf_request("POST",
    f"/accounts/{ACCOUNT}/pages/projects/{PROJECT}/deployments/{dep_id}/finalize",
    body=b"{}",
    content_type="application/json")
print("Finalize:", fin.get("success"), fin.get("errors"))

dep_url = fin.get("result", {}).get("url", "")
print(f"\n🚀 Preview: {dep_url}")
print(f"🌐 Production: https://qinche-tts.pages.dev")
