"""Clone official source, preserve incomplete downloads, and record git identity."""
from datetime import datetime
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "ultralytics_src"
LOGS = PROJECT / "workspace" / "logs"


def git(*args):
    return subprocess.run(["git", "-C", str(SOURCE), *args], capture_output=True, encoding="utf-8", errors="replace")


def main():
    if not (SOURCE / ".git").exists() or git("rev-parse", "--verify", "HEAD").returncode:
        if SOURCE.exists() and any(SOURCE.iterdir()):
            if {p.name for p in SOURCE.iterdir()} != {".git"}:
                raise RuntimeError("Source directory contains existing files; refusing to relocate it.")
            destination = LOGS / ("incomplete_clone_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            assert SOURCE.resolve().is_relative_to(PROJECT.resolve())
            assert destination.resolve().is_relative_to(PROJECT.resolve())
            assert SOURCE.resolve() != PROJECT.resolve() and not destination.exists()
            print(f"Preserving incomplete clone: {SOURCE.resolve()} -> {destination.resolve()}", flush=True)
            SOURCE.rename(destination)
        command = ["git", "-c", "http.version=HTTP/1.1", "-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=30",
                   "clone", "--depth", "1", "--progress", "https://github.com/ultralytics/ultralytics.git", str(SOURCE)]
        with (LOGS / "git_clone.log").open("w", encoding="utf-8") as log:
            log.write(subprocess.list2cmdline(command) + "\n")
            log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
    sections = []
    for args in (("branch", "--show-current"), ("rev-parse", "HEAD"), ("remote", "-v"),
                 ("status", "--short", "--branch"), ("status", "--porcelain"), ("rev-parse", "--is-shallow-repository")):
        p = git(*args)
        p.check_returncode()
        sections.append("> git " + " ".join(args) + "\n" + p.stdout + p.stderr)
    report = "\n".join(sections)
    (LOGS / "git_info.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
