#!/usr/bin/env python3
"""Upload dist/ to Cloudflare Pages via Direct Upload API."""
import os, sys, hashlib, mimetypes, json, ssl
import urllib.request

TOKEN   = "nTcN1cTgOeD5M0z5iKa_9KlXPTPPYeBpyaqJ4OJI"
ACCOUNT = "040b706a29a1319752dc78889ef4b3f5"
PROJECT = "qinche-tts"
DIST    = os.path.join(os.path.dirname(__file__), "dist")
CTX     = ssl.create_default_context()

def cf(method, path, body=None, ctype=None):
    url = f"https://api.cloudflare.com/client/v4{path}"
    hdrs = {"Authorization": f"Bearer {TOKEN}"}
    if ctype:
        hdrs["Content-Type"] = ctype
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, context=CTX) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        txt = e.read()
        try: return json.loads(txt)
        except: return {"success": False, "raw": txt.decode()}

# 1. Collect files
files = {}
for root, _, fnames in os.walk(DIST):
    for fname in fnames:
        fpath = os.path.join(root, fname)
        rel = "/" + os.path.relpath(fpath, DIST).replace(os.sep, "/")
        with open(fpath, "rb") as f:
            content = f.read()
        h = hashlib.sha256(content).hexdigest()
        files[rel] = {
            "path": fpath, "hash": h, "content": content,
            "mime": mimetypes.guess_type(rel)[0] or "application/octet-stream"
        }
print(f"Files: {len(files)}")

# 2. Create deployment with manifest
BOUNDARY = "CFPagesDeploy"
manifest = {rel: info["hash"] for rel, info in files.items()}
body = (
    f"--{BOUNDARY}\r\n"
    f'Content-Disposition: form-data; name="manifest"\r\n\r\n'
    + json.dumps(manifest)
    + f"\r\n--{BOUNDARY}--\r\n"
).encode()

resp = cf("POST",
    f"/accounts/{ACCOUNT}/pages/projects/{PROJECT}/deployments",
    body=body,
    ctype=f"multipart/form-data; boundary={BOUNDARY}")
print("Create:", resp.get("success"), resp.get("errors", resp.get("raw","")))
if not resp.get("success"):
    sys.exit(1)
dep_id = resp["result"]["id"]
print(f"Deployment: {dep_id}")

# 3. Upload files — use PUT with base64 encoded payload via form
print("Uploading...")
for rel, info in files.items():
    B = f"CFFile{info['hash'][:6]}"
    fbody = (
        f"--{B}\r\n"
        f'Content-Disposition: form-data; name="{info["hash"]}"; filename="{info["hash"]}"\r\n'
        f'Content-Type: {info["mime"]}\r\n\r\n'
    ).encode() + info["content"] + f"\r\n--{B}--\r\n".encode()

    r = cf("PUT",
        f"/accounts/{ACCOUNT}/pages/projects/{PROJECT}/deployments/{dep_id}/files",
        body=fbody,
        ctype=f"multipart/form-data; boundary={B}")
    ok = r.get("success", False)
    print(f"  {'✓' if ok else '✗'} {rel}" + ("" if ok else f"  {r.get('errors', r.get('raw',''))}"))

# 4. Finalize
print("Finalizing...")
fin = cf("POST",
    f"/accounts/{ACCOUNT}/pages/projects/{PROJECT}/deployments/{dep_id}/finalize",
    body=b"", ctype="application/json")
print("Finalize:", fin.get("success"), fin.get("errors",""))
dep_url = (fin.get("result") or {}).get("url", "")
print(f"\n🚀 Preview:    {dep_url}")
print(f"🌐 Production: https://qinche-tts.pages.dev")
