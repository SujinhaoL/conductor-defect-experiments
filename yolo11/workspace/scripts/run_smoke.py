"""Run the fixed five-epoch smoke test and retain complete stdout and warnings."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = project / "workspace" / "logs" / f"smoke_console_{stamp}.log"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    command = [str(project / ".venv" / "Scripts" / "python.exe"), "-X", "utf8", "-u",
               str(project / "workspace" / "scripts" / "train_yolo11.py")]
    print(f"Console log: {logfile}", flush=True)
    with logfile.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        log.write(f"\nPROCESS_EXIT_CODE={code}\n")
    raise SystemExit(code)
