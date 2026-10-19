"""Prepare Microsoft's signed bootstrapper; never execute an installer."""
import argparse
import ctypes
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
DEST = PROJECT / "workspace/installers/vs_buildtools_2022.exe"
MANIFEST = DEST.with_suffix(".manifest.json")
SOURCE = "https://aka.ms/vs/17/release/vs_buildtools.exe"


def digest(file):
    h = hashlib.sha256()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(file):
    # Native read-only Authenticode verification; never runs the downloaded EXE.
    env = os.environ.copy()
    env["FRCNN_SIGNATURE_TARGET"] = str(file)
    ps = r"""
$ErrorActionPreference = 'Stop'
$s = Get-AuthenticodeSignature -LiteralPath $env:FRCNN_SIGNATURE_TARGET
$v = (Get-Item -LiteralPath $env:FRCNN_SIGNATURE_TARGET).VersionInfo
[pscustomobject]@{Status=$s.Status.ToString();Subject=$s.SignerCertificate.Subject;FileVersion=$v.FileVersion;ProductVersion=$v.ProductVersion} | ConvertTo-Json -Compress
"""
    done = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                          env=env, capture_output=True, timeout=60,
                          creationflags=subprocess.CREATE_NO_WINDOW)
    if done.returncode:
        raise RuntimeError("Windows Authenticode verification failed")
    info = json.loads(done.stdout.decode("utf-8-sig", errors="replace"))
    if info["Status"] != "Valid" or "O=Microsoft Corporation" not in (info["Subject"] or ""):
        raise RuntimeError("Installer signature is not a valid Microsoft signature: " + str(info))
    if not str(info["FileVersion"]).startswith("17.") or "2022" not in str(info["ProductVersion"]):
        raise RuntimeError("Expected Visual Studio 2022 / major version 17: " + str(info))
    return info


def verify():
    if not DEST.is_file() or not MANIFEST.is_file():
        raise RuntimeError("Prepared installer or manifest is missing")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if digest(DEST) != manifest["sha256"]:
        raise RuntimeError("Installer SHA256 differs from prepared manifest")
    info = signature(DEST)
    print("MICROSOFT_SIGNATURE_VALID")
    print("INSTALLER_VERSION=" + str(info["FileVersion"]))
    print("INSTALLER_SHA256=" + manifest["sha256"])


def prepare():
    if DEST.exists() or MANIFEST.exists():
        verify()
        return
    from pip._vendor import requests
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temp = DEST.with_suffix(".download.exe")
    evidence = {"timestamp_asia_shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "source_url": SOURCE, "mmcv_index_probe": {},
                "c_drive_free_gib": round(shutil.disk_usage("C:/").free / 2**30, 2)}
    with requests.Session() as session:
        session.trust_env = False
        index = "https://download.openmmlab.com/mmcv/dist/cu128/torch2.7.0/index.html"
        try:
            response = session.get(index, timeout=(10, 15))
            evidence["mmcv_index_probe"] = {"url": index, "http_status": response.status_code,
                                            "matching_cp310_windows_mmcv_2_1_wheel": "mmcv-2.1.0-cp310-cp310-win_amd64.whl" in response.text}
        except Exception as exc:
            evidence["mmcv_index_probe"] = {"url": index, "error_type": type(exc).__name__}
        with session.get(SOURCE, stream=True, timeout=(10, 30)) as response:
            response.raise_for_status()
            evidence["resolved_url"] = response.url
            with temp.open("wb") as f:
                for block in response.iter_content(1024 * 1024):
                    f.write(block)
    with temp.open("rb") as f:
        if f.read(2) != b"MZ":
            raise RuntimeError("Downloaded file is not a Windows executable")
    evidence["signature"] = signature(temp)
    evidence["size_bytes"] = temp.stat().st_size
    evidence["sha256"] = digest(temp)
    # Rename only the exact temporary file created above, within installers/.
    temp.rename(DEST)
    MANIFEST.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    print("PREPARED_INSTALLER=" + str(DEST))
    print("NO_INSTALLATION_PERFORMED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-installer", action="store_true")
    parser.add_argument("--require-admin", action="store_true")
    args = parser.parse_args()
    if args.require_admin:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("ERROR: Open Command Prompt as administrator, then run the same command.")
            return 1
        print("ADMINISTRATOR_CHECK_PASS")
    elif args.verify_installer:
        verify()
    else:
        prepare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
