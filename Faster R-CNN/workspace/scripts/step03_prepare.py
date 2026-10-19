"""Download official STEP03 assets; no install, model execution, or training."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "mmdetection"
COMMIT = "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a"
WEIGHT_URL = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
WEIGHT = ROOT / "workspace/weights" / WEIGHT_URL.rsplit("/", 1)[1]
WHEELS = ROOT / "workspace/wheels/step03_dependencies"
PINS = {"pycocotools": "2.0.10", "scipy": "1.15.3", "shapely": "2.1.1",
        "terminaltables": "3.1.10", "tqdm": "4.67.1", "colorama": "0.4.6"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch_ranges(session, url, part, total):
    """Read four bounded ranges concurrently, then assemble and hash the file."""
    from pip._vendor import requests
    chunk_size = 4 * 1024 * 1024
    ranges = [(start, min(start + chunk_size, total) - 1) for start in range(0, total, chunk_size)]

    def read_range(bounds):
        start, end = bounds
        target = part.with_name(part.name + f".{start}")
        if target.is_file() and target.stat().st_size == end - start + 1:
            return target
        for attempt in range(3):
            try:
                with requests.Session() as client:
                    client.trust_env = False
                    response = client.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 60))
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise RuntimeError("Server returned an unexpected HTTP range")
                    if len(response.content) != end - start + 1:
                        raise RuntimeError("Incomplete HTTP range")
                    target.write_bytes(response.content)
                return target
            except Exception:
                if attempt == 2:
                    raise
    with ThreadPoolExecutor(max_workers=4) as executor:
        pieces = list(executor.map(read_range, ranges))
    with part.open("wb") as output:
        for piece in pieces:
            output.write(piece.read_bytes())
    for piece in pieces:
        piece.unlink()
    print(f"OFFICIAL_RANGE_DOWNLOAD_COMPLETE={part.name}: {total} bytes", flush=True)


def fetch(url, path, expected_hash=None, expected_size=None):
    from pip._vendor import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        actual = digest(path)
        if expected_hash and actual == expected_hash:
            return actual
        if path == WEIGHT and actual.startswith("047c8118"):
            return actual
        raise RuntimeError("Existing file failed integrity check: " + str(path))
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.Session() as session:
                session.trust_env = False
                if expected_size and expected_size > 16 * 1024 * 1024:
                    fetch_ranges(session, url, part, expected_size)
                else:
                    with session.get(url, stream=True, timeout=(20, 90)) as response:
                        response.raise_for_status()
                        total = int(response.headers.get("Content-Length", 0))
                        done, last = 0, time.monotonic()
                        with part.open("wb") as output:
                            for block in response.iter_content(1024 * 1024):
                                output.write(block)
                                done += len(block)
                                if time.monotonic() - last > 25:
                                    print(f"DOWNLOADING {path.name}: {done}/{total} bytes", flush=True)
                                    last = time.monotonic()
                        if total and done != total:
                            raise RuntimeError("Incomplete download")
            actual = digest(part)
            if expected_hash and actual != expected_hash:
                raise RuntimeError("SHA256 mismatch: " + path.name)
            if path == WEIGHT and not actual.startswith("047c8118"):
                raise RuntimeError("Checkpoint SHA256 does not match published filename prefix")
            part.replace(path)
            print("DOWNLOAD_HASH_PASS=" + path.name, flush=True)
            return actual
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def prepare_wheel(item):
    from pip._vendor import requests
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    name, version = item
    supported = set(sys_tags())
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(f"https://pypi.org/pypi/{name}/{version}/json", timeout=(20, 60))
        response.raise_for_status()
        matches = [x for x in response.json()["urls"] if x["filename"].endswith(".whl")
                   and supported.intersection(parse_wheel_filename(x["filename"])[3])]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one compatible official wheel for {name}: {len(matches)}")
    published = matches[0]
    path = WHEELS / published["filename"]
    actual = fetch(published["url"], path, published["digests"]["sha256"], published["size"])
    return {"name": name, "version": version, "filename": path.name, "url": published["url"],
            "sha256": actual, "size_bytes": path.stat().st_size}


def main():
    if Path(sys.prefix).resolve() != ROOT / ".venv":
        raise RuntimeError("Use only this project's dedicated Python")
    git = ["git", "-c", "safe.directory=" + SOURCE.as_posix()]
    head = subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if head != COMMIT:
        raise RuntimeError("Unexpected MMDetection commit")
    if subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"], cwd=SOURCE):
        raise RuntimeError("Official MMDetection tracked files were changed")
    WHEELS.mkdir(parents=True, exist_ok=True)
    constraints = ROOT / "workspace/configs/step03_base_constraints.txt"
    if not constraints.exists():
        baseline = sorted({f'{d.metadata["Name"]}=={d.version}' for d in md.distributions()})
        constraints.write_text("\n".join(baseline) + "\n", encoding="utf-8")
    # A separate session per worker avoids shared HTTP session state.
    with ThreadPoolExecutor(max_workers=3) as executor:
        checkpoint_future = executor.submit(fetch, WEIGHT_URL, WEIGHT, None, 167287506)
        futures = [executor.submit(prepare_wheel, item) for item in PINS.items()]
        records = [f.result() for f in futures]
        checkpoint_hash = checkpoint_future.result()
    lock = ROOT / "workspace/configs/step03_requirements.lock.txt"
    lock.write_text("# Official PyPI wheels verified against published SHA256.\n" + "\n".join(
        f'{r["name"]}=={r["version"]} --hash=sha256:{r["sha256"]}'
        for r in sorted(records, key=lambda r: r["name"])) + "\n", encoding="utf-8")
    config = SOURCE / "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
    manifest = {"prepared_utc": datetime.now(timezone.utc).isoformat(),
                "mmdet_version": "3.3.0", "mmdet_commit": COMMIT,
                "remote": "https://github.com/open-mmlab/mmdetection.git",
                "source": str(SOURCE), "config": str(config), "config_sha256": digest(config),
                "demo_image": str(SOURCE / "demo/demo.jpg"),
                "demo_image_sha256": digest(SOURCE / "demo/demo.jpg"),
                "checkpoint": {"path": str(WEIGHT), "url": WEIGHT_URL, "sha256": checkpoint_hash,
                               "size_bytes": WEIGHT.stat().st_size,
                               "published_filename_sha256_prefix_match": True},
                "dependency_wheels": records, "lock_sha256": digest(lock),
                "base_constraints_sha256": digest(constraints), "training_started": False}
    (ROOT / "workspace/configs/step03_preparation.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STEP03_PREPARATION_PASS", flush=True)
    print("CHECKPOINT_SHA256=" + checkpoint_hash, flush=True)


if __name__ == "__main__":
    main()
