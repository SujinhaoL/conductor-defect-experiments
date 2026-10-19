"""User-run MMCV wheel build/install; never invoke a detector or training."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/src/mmcv"
COMMIT = "57c4e25e06e2d4f8a9357c84bcd24089a284dc88"
# Trust only this official checkout for each command, without changing global Git config.
GIT = ["git", "-c", "safe.directory=" + SOURCE.as_posix()]
LOCK = ROOT / "workspace/configs/step02c_requirements.lock.txt"
MANIFEST = ROOT / "workspace/configs/step02c_preparation.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_base():
    check_identity = runpy.run_path(str(ROOT / "workspace/scripts/step01_check_cuda.py"))["check_identity"]
    identity = check_identity()
    constraints = ROOT / "workspace/configs/step02c_base_constraints.txt"
    for line in constraints.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        require(metadata.version(name) == version, "Previously accepted package changed: " + name)
    print("ENVIRONMENT_IDENTITY_AND_BASE_VERSIONS_PASS", flush=True)
    return identity


def check_preparation():
    check_base()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(data["mmcv_commit"] == COMMIT, "Preparation commit mismatch")
    head = subprocess.check_output(GIT + ["rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    require(head == COMMIT, "Official MMCV checkout changed")
    changes = subprocess.check_output(GIT + ["diff", "HEAD", "--name-only"], cwd=SOURCE)
    require(not changes, "Official MMCV tracked source was changed")
    require(digest(LOCK) == data["lock_sha256"], "Dependency lock changed")
    for record in data["dependency_wheels"]:
        filename = record["filename"]
        require(Path(filename).name == filename, "Invalid wheel filename")
        wheel = ROOT / "workspace/wheels/step02c_dependencies" / filename
        require(digest(wheel) == record["sha256"], "Dependency wheel hash mismatch: " + filename)
    print("SOURCE_COMMIT_AND_DEPENDENCY_HASHES_PASS", flush=True)
    return data


def run(command, cwd, env, log):
    print("COMMAND=" + subprocess.list2cmdline([str(x) for x in command]), flush=True)
    with log.open("wb") as output:
        child = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for line in iter(child.stdout.readline, b""):
                output.write(line)
                output.flush()
                print(line.decode("utf-8", errors="replace"), end="", flush=True)
            code = child.wait()
        except BaseException:
            # Stop only the process tree started by this runner, never other jobs.
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
            child.wait()
            raise
    require(code == 0, f"Command exited {code}; inspect {log}")


def create_alias():
    occupied = ctypes.windll.kernel32.GetLogicalDrives()
    for letter in "RSTUVWXYZQP":
        if occupied & (1 << (ord(letter) - ord("A"))):
            continue
        drive = letter + ":"
        result = subprocess.run(["subst", drive, str(ROOT)], capture_output=True)
        if result.returncode:
            continue
        if Path(drive + "\\").resolve() != ROOT:
            subprocess.run(["subst", drive, "/D"], check=True)
            raise RuntimeError("Temporary drive did not resolve to this project")
        return drive
    raise RuntimeError("No free temporary drive letter; no files were moved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--check-preparation", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.check_preparation:
        check_preparation()
        drive = create_alias()
        try:
            # Smoke-check the alias and Python identity without installing or compiling.
            alias = Path(drive + "\\")
            subprocess.run([str(alias / ".venv/Scripts/python.exe"), "-I", "-X", "utf8",
                            str(alias / "workspace/scripts/step01_check_cuda.py"), "--identity-only"], check=True)
            print("TEMPORARY_ALIAS_IDENTITY_PASS")
        finally:
            require(Path(drive + "\\").resolve() == ROOT, "Temporary alias changed externally")
            subprocess.run(["subst", drive, "/D"], check=True)
        print("STEP02C_PREPARATION_CHECK_PASS")
        return 0
    require(args.log_dir is not None, "--log-dir is required")
    logs = args.log_dir.resolve()
    require(logs.is_relative_to(ROOT / "workspace/logs"), "Logs must stay in this workspace")
    logs.mkdir(parents=True, exist_ok=True)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "FAIL",
              "mmcv_commit": COMMIT, "source": str(SOURCE), "training_started": False}
    drive = None
    start = time.monotonic()
    try:
        check_base()
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                   PIP_CONFIG_FILE="NUL", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
                   MPLBACKEND="Agg", MPLCONFIGDIR=str(ROOT / "workspace/cache/matplotlib"))
        if args.verify_only:
            python = Path(sys.executable)
            child_logs = logs
            child_root = ROOT
        else:
            data = check_preparation()
            try:
                installed = metadata.version("mmcv")
            except metadata.PackageNotFoundError:
                installed = None
            require(installed is None, "MMCV is already installed; use --verify-only instead of rebuilding")
            require(os.environ.get("VCToolsVersion", "").startswith("14.29"), "MSVC v142 is not selected")
            require(os.environ.get("WindowsSDKVersion") == "10.0.19041.0\\", "Wrong Windows SDK")
            required = {"TORCH_CUDA_ARCH_LIST": "12.0", "MAX_JOBS": "4", "MMCV_WITH_OPS": "1",
                        "FORCE_CUDA": "1", "DISTUTILS_USE_SDK": "1"}
            for key, value in required.items():
                require(env.get(key) == value, "Unexpected build setting: " + key)
            cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
            require(Path(env.get("CUDA_HOME", "")).resolve() == cuda_root, "Wrong CUDA_HOME")
            cuda_metadata = json.loads((cuda_root / "version.json").read_text(encoding="utf-8"))
            require(cuda_metadata["cuda"]["version"] == "12.8.1" and cuda_metadata["cuda_nvcc"]["version"] == "12.8.93", "Unexpected Toolkit components")
            drive = create_alias()
            child_root = Path(drive + "\\")
            python = child_root / ".venv/Scripts/python.exe"
            child_logs = child_root / logs.relative_to(ROOT)
            temporary = child_root / "workspace/cache/mmcv_build_temp"
            temporary.mkdir(parents=True, exist_ok=True)
            env.update(TEMP=str(temporary), TMP=str(temporary))
            env["PATH"] = str(child_root / ".venv/Scripts") + os.pathsep + env["PATH"]
            env["VIRTUAL_ENV"] = str(child_root / ".venv")
            print("TEMPORARY_BUILD_ALIAS=" + drive + " -> " + str(ROOT), flush=True)
            report["temporary_alias"] = drive
            report["build_environment"] = {key: env.get(key) for key in [*required, "CXX", "CUDA_HOME", "VCToolsVersion", "WindowsSDKVersion", "TEMP"]}
            report["dependency_lock_sha256"] = data["lock_sha256"]
            # Validate the alias still selects the exact same environment before installation.
            run([python, "-I", "-X", "utf8", child_root / "workspace/scripts/step01_check_cuda.py", "--identity-only"],
                child_root, env, logs / "alias_identity.txt")
            pip = [python, "-I", "-X", "utf8", "-m", "pip", "--isolated", "--disable-pip-version-check"]
            print("[1/4] Installing pinned dependencies from verified local wheels.", flush=True)
            run(pip + ["install", "--no-index", "--no-input", "--require-hashes", "--only-binary=:all:",
                       "--find-links", child_root / "workspace/wheels/step02c_dependencies",
                       "--report", child_logs / "dependencies_install.json", "-r", child_root / LOCK.relative_to(ROOT)],
                child_root, env, logs / "dependencies_install.txt")
            print("[2/4] Building official MMCV CUDA wheel. This can take tens of minutes; no training.", flush=True)
            built = child_logs / "wheels"
            built.mkdir(exist_ok=True)
            build_start = time.monotonic()
            run(pip + ["wheel", ".", "--verbose", "--use-pep517", "--no-build-isolation", "--no-deps", "--no-index", "--wheel-dir", built],
                child_root / "workspace/src/mmcv", env, logs / "mmcv_build.txt")
            report["build_seconds"] = round(time.monotonic() - build_start, 2)
            wheels = list(built.glob("mmcv-2.1.0-*.whl"))
            require(len(wheels) == 1, "Expected exactly one MMCV 2.1.0 wheel")
            wheel = wheels[0]
            report["built_wheel"] = {"path": str(wheel.resolve()), "sha256": digest(wheel), "size_bytes": wheel.stat().st_size}
            print("MMCV_WHEEL_SHA256=" + report["built_wheel"]["sha256"], flush=True)
            print("[3/4] Installing the locally built wheel into this project's environment.", flush=True)
            # Record the permanent path in pip's direct_url metadata, not the temporary alias.
            run(pip + ["install", "--no-index", "--no-deps", "--report", child_logs / "mmcv_install.json", wheel.resolve()],
                child_root, env, logs / "mmcv_install.txt")
        pip = [python, "-I", "-X", "utf8", "-m", "pip", "--isolated", "--disable-pip-version-check"]
        run(pip + ["check"], child_root, env, logs / "pip_check.txt")
        # Saved before the operator check, so a failed native import remains reproducible.
        frozen = subprocess.check_output(pip + ["freeze", "--all"], cwd=child_root, env=env)
        (logs / "requirements_freeze.txt").write_bytes(frozen)
        check_base()
        print("[4/4] Validating MMCV NMS and RoIAlign on synthetic tensors, not training.", flush=True)
        run([python, "-I", "-X", "utf8", child_root / "workspace/scripts/step02c_check_mmcv.py", "--output-dir", child_logs],
            child_root, env, logs / "mmcv_ops_check.txt")
        require(not subprocess.check_output(GIT + ["diff", "HEAD", "--name-only"], cwd=SOURCE), "Tracked MMCV source changed during build")
        report["status"] = "PASS"
    except BaseException:
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        if drive:
            # Remove only the mapping this invocation created, and only if it still points here.
            if Path(drive + "\\").resolve() == ROOT:
                result = subprocess.run(["subst", drive, "/D"], capture_output=True)
                report["temporary_alias_removed"] = result.returncode == 0
                if result.returncode == 0:
                    print("TEMPORARY_BUILD_ALIAS_REMOVED=" + drive, flush=True)
                else:
                    report["status"] = "FAIL"
                    print("Could not remove temporary drive mapping " + drive, flush=True)
            else:
                report["status"] = "FAIL"
                print("Temporary mapping changed externally; it was left untouched.", flush=True)
        report["elapsed_seconds"] = round(time.monotonic() - start, 2)
        (logs / "build_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["status"] == "PASS":
        print("STEP02C_MMCV_PASS", flush=True)
        return 0
    print("STEP02C_MMCV_FAIL", flush=True)
    print("Send complete terminal output and LOG_DIR. Do not continue to MMDetection yet.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
