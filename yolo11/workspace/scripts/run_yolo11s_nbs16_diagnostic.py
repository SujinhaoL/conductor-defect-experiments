"""Run Y11-DIAG-002 and retain complete console output."""

from datetime import datetime
import os
from pathlib import Path
import subprocess


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = project / "workspace" / "logs" / f"yolo11s_nbs16_console_{stamp}.log"
    environment = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONUNBUFFERED="1",
        NBS16_CONSOLE_LOG=str(logfile),
    )
    command = [
        str(project / ".venv" / "Scripts" / "python.exe"),
        "-X", "utf8", "-u",
        str(project / "workspace" / "scripts" / "train_yolo11_nbs16_diagnostic.py"),
    ]
    print(f"Console log: {logfile}", flush=True)
    with logfile.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            cwd=project,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        log.write(f"\nPROCESS_EXIT_CODE={code}\n")
    raise SystemExit(code)
