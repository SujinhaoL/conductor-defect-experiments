"""Exercise synthetic audit events without performing any dataset mutation."""
import json
import os
import sys
from readonly_dataset import DATASET, WORKSPACE, install_write_guard

if __name__ == "__main__":
    install_write_guard()
    path = str(DATASET / "guard_probe_never_created.tmp")
    events = [("open", (path, "w", os.O_WRONLY | os.O_CREAT)),
              ("open", (path, None, os.O_RDWR)),
              ("os.remove", (path, -1)), ("os.mkdir", (path, 511, -1)),
              ("os.rename", (path, str(WORKSPACE / "probe.tmp"), -1, -1)),
              ("os.rename", (str(WORKSPACE / "probe.tmp"), path, -1, -1)),
              ("os.utime", (path, None, None, -1)), ("os.chmod", (path, 438, -1))]
    passed = []
    for event, args in events:
        try:
            sys.audit(event, *args)
        except PermissionError:
            passed.append(event)
        else:
            raise RuntimeError(f"Write guard did not block {event}")
    sys.audit("open", path, "r", os.O_RDONLY)
    result = {"passed": True, "synthetic_write_events_blocked": len(passed), "read_allowed": True,
              "note": "Synthetic events only; no filesystem mutation was attempted."}
    (WORKSPACE / "logs" / "write_guard_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)
