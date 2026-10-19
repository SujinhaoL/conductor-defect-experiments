"""Download and authenticate CUDA Toolkit 12.8 Update 1; never install it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[2]
DEST = PROJECT / "workspace/installers/cuda_12.8.1_572.61_windows.exe"
MANIFEST = DEST.with_suffix(".manifest.json")
SOURCE = "https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/" + DEST.name
CHECKSUM_URL = "https://developer.download.nvidia.com/compute/cuda/12.8.1/docs/sidebar/md5sum.txt"
EXPECTED_MD5 = "05c59d04ac6c63686c6a0498e472c1e7"


def hashes(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def signature(path):
    # PowerShell is used only for Windows' native signature API, not installation.
    env = os.environ.copy()
    env["FRCNN_SIGNATURE_TARGET"] = str(path)
    command = r"""
$ErrorActionPreference = 'Stop'
$s = Get-AuthenticodeSignature -LiteralPath $env:FRCNN_SIGNATURE_TARGET
$v = (Get-Item -LiteralPath $env:FRCNN_SIGNATURE_TARGET).VersionInfo
[pscustomobject]@{Status=$s.Status.ToString();Subject=$s.SignerCertificate.Subject;FileVersion=$v.FileVersion;ProductVersion=$v.ProductVersion} | ConvertTo-Json -Compress
"""
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            env=env, capture_output=True, timeout=60,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError("Windows Authenticode verification failed")
    info = json.loads(result.stdout.decode("utf-8-sig", errors="replace"))
    if info["Status"] != "Valid" or "NVIDIA Corporation" not in (info["Subject"] or ""):
        raise RuntimeError("Expected a valid NVIDIA signature: " + str(info))
    return info


def verify():
    if not DEST.is_file() or not MANIFEST.is_file():
        raise RuntimeError("Prepared CUDA installer or manifest is missing")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = hashes(DEST)
    if actual["md5"] != EXPECTED_MD5 or actual["sha256"] != manifest["sha256"]:
        raise RuntimeError("CUDA installer checksum mismatch")
    signature(DEST)
    print("NVIDIA_SIGNATURE_VALID")
    print("OFFICIAL_MD5_MATCH=" + actual["md5"])
    print("INSTALLER_SHA256=" + actual["sha256"])
    print("INSTALLER_BYTES=" + str(DEST.stat().st_size))


def prepare(probe_only=False):
    from pip._vendor import requests
    if DEST.exists() or MANIFEST.exists():
        verify()
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = DEST.with_suffix(".download.exe")
    with requests.Session() as session:
        session.trust_env = False  # Direct HTTPS; no persistent proxy changes.
        checksum = session.get(CHECKSUM_URL, timeout=(15, 60))
        checksum.raise_for_status()
        entries = dict((line.split()[1], line.split()[0]) for line in checksum.text.splitlines() if line.strip())
        if entries.get(DEST.name) != EXPECTED_MD5:
            raise RuntimeError("Official checksum list does not match the pinned installer")
        head = session.head(SOURCE, allow_redirects=True, timeout=(15, 60))
        head.raise_for_status()
        size = int(head.headers["Content-Length"])
        if size < 1024**3:
            raise RuntimeError("Unexpectedly small local CUDA installer: " + str(size))
        print("SOURCE=" + SOURCE, flush=True)
        print("INSTALLER_BYTES=" + str(size), flush=True)
        print("C_DRIVE_FREE_GIB=" + str(round(shutil.disk_usage("C:/").free / 2**30, 2)), flush=True)
        if probe_only:
            print("PROBE_ONLY_NO_DOWNLOAD_NO_INSTALL")
            return
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > size:
            raise RuntimeError("Partial download is larger than the official installer")
        if shutil.disk_usage(DEST.parent).free < size - offset + 1024**3:
            raise RuntimeError("Insufficient free disk space for download")
        if offset < size:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session.get(SOURCE, headers=headers, stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise RuntimeError("Server did not honor the resume range; keeping partial download")
                last_report = time.monotonic()
                with temporary.open("ab" if offset else "wb") as stream:
                    for block in response.iter_content(4 * 1024 * 1024):
                        stream.write(block)
                        offset += len(block)
                        if time.monotonic() - last_report >= 15:
                            print(f"DOWNLOAD_BYTES={offset}/{size} ({100 * offset / size:.1f}%)", flush=True)
                            last_report = time.monotonic()
        if temporary.stat().st_size != size:
            raise RuntimeError("Downloaded size differs from Content-Length")
        actual = hashes(temporary)
        if actual["md5"] != EXPECTED_MD5:
            raise RuntimeError("Downloaded installer does not match NVIDIA's MD5; file retained for diagnosis")
        verified_signature = signature(temporary)
        evidence = {"prepared_utc": datetime.now(timezone.utc).isoformat(),
                    "source_url": SOURCE, "resolved_url": head.url,
                    "official_checksum_url": CHECKSUM_URL, "size_bytes": size,
                    **actual, "signature": verified_signature,
                    "toolkit_version": "12.8.1", "bundled_driver_not_to_be_installed": "572.61"}
        temporary.rename(DEST)
        MANIFEST.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        (DEST.parent / "cuda_12.8.1_official_md5sum.txt").write_text(checksum.text, encoding="utf-8")
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        print("CUDA_INSTALLER_PREPARED_NO_INSTALLATION")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify-installer", action="store_true")
    group.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.verify_installer:
        verify()
    else:
        prepare(probe_only=args.probe)
