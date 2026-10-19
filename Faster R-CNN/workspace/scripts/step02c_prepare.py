"""Prepare official dependency wheels and provenance, without installing or building."""
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/src/mmcv"
COMMIT = "57c4e25e06e2d4f8a9357c84bcd24089a284dc88"
WHEELS = ROOT / "workspace/wheels/step02c_dependencies"
LOCK = ROOT / "workspace/configs/step02c_requirements.lock.txt"
MANIFEST = ROOT / "workspace/configs/step02c_preparation.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if Path(sys.prefix).resolve() != ROOT / ".venv":
        raise RuntimeError("Use only this project's dedicated Python")
    git = ["git", "-c", "safe.directory=" + SOURCE.as_posix()]
    head = subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if head != COMMIT:
        raise RuntimeError("Unexpected MMCV commit")
    if subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"], cwd=SOURCE):
        raise RuntimeError("Official MMCV source has tracked changes")
    WHEELS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in list(env):
        if key.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}:
            del env[key]
    env.update(NO_PROXY="*", PIP_CONFIG_FILE="NUL", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    # Pin every package already accepted in STEP01; do not alter that environment.
    freeze = subprocess.check_output([sys.executable, "-I", "-X", "utf8", "-m", "pip", "--isolated", "freeze", "--all"], env=env, text=True, encoding="utf-8")
    constraints = ROOT / "workspace/configs/step02c_base_constraints.txt"
    constraints.write_text(freeze, encoding="utf-8")
    command = [sys.executable, "-I", "-X", "utf8", "-m", "pip", "--isolated", "--disable-pip-version-check",
               "--log", str(ROOT / "workspace/logs/step02c_dependency_download.log"),
               "download", "--only-binary=:all:", "--index-url", "https://pypi.org/simple", "--retries", "3", "--timeout", "60",
               "--dest", str(WHEELS), "-r", str(ROOT / "workspace/configs/step02c_requirements.in"), "-c", str(constraints)]
    subprocess.run(command, env=env, check=True)
    from pip._vendor import requests
    records = []
    seen = set()
    with requests.Session() as session:
        session.trust_env = False
        for wheel in sorted(WHEELS.glob("*.whl")):
            with zipfile.ZipFile(wheel) as archive:
                names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
                if len(names) != 1:
                    raise RuntimeError("Expected exactly one wheel METADATA")
                metadata = BytesParser().parsebytes(archive.read(names[0]))
            name = re.sub(r"[-_.]+", "-", metadata["Name"]).lower()
            version = metadata["Version"]
            if name in seen:
                raise RuntimeError("Multiple wheels for one dependency; inspect before continuing")
            seen.add(name)
            response = session.get(f"https://pypi.org/pypi/{name}/{version}/json", timeout=(15, 60))
            response.raise_for_status()
            published = next(x for x in response.json()["urls"] if x["filename"] == wheel.name)
            actual = sha256(wheel)
            if actual != published["digests"]["sha256"]:
                raise RuntimeError("PyPI SHA256 mismatch: " + wheel.name)
            records.append({"name": name, "version": version, "filename": wheel.name,
                            "sha256": actual, "url": published["url"], "size_bytes": wheel.stat().st_size})
            print("OFFICIAL_PYPI_HASH_PASS=" + wheel.name, flush=True)
    lines = ["# Prepared from official PyPI; install offline with --require-hashes."]
    lines += [f'{r["name"]}=={r["version"]} --hash=sha256:{r["sha256"]}' for r in sorted(records, key=lambda x: x["name"])]
    LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    evidence = {"prepared_utc": datetime.now(timezone.utc).isoformat(), "mmcv_version": "2.1.0",
                "mmcv_commit": COMMIT, "mmcv_remote": "https://github.com/open-mmlab/mmcv.git",
                "source_directory": str(SOURCE), "source_tracked_changes": False,
                "lock_sha256": sha256(LOCK), "dependency_wheels": records,
                "scope": "Download and integrity checks only; no installation, compilation, or training."}
    MANIFEST.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STEP02C_PREPARATION_PASS")
    print("DEPENDENCY_WHEELS=" + str(len(records)))
    print("MMCV_COMMIT=" + COMMIT)


if __name__ == "__main__":
    main()
