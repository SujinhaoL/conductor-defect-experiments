"""Diagnose package-index connectivity without installing or changing OS settings."""
import json
import os
from pathlib import Path
import platform
import sys
import urllib.parse
import urllib.request
import winreg
from datetime import datetime, timezone, timedelta


def safe_proxy(value):
    if not value:
        return None
    parsed = urllib.parse.urlsplit(value if "://" in value else "http://" + value)
    try:
        port = parsed.port
    except ValueError:
        port = "invalid"
    return {"scheme": parsed.scheme, "host": parsed.hostname, "port": port,
            "credentials_present": bool(parsed.username or parsed.password)}


def main():
    from pip._vendor import requests
    result = {"python": platform.python_version(), "executable": sys.executable,
              "environment_proxies": {}, "discovered_proxies": {}, "requests": []}
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if os.environ.get(k):
            result["environment_proxies"][k] = safe_proxy(os.environ[k])
    result["no_proxy_set"] = bool(os.environ.get("NO_PROXY") or os.environ.get("no_proxy"))
    for k, v in urllib.request.getproxies().items():
        result["discovered_proxies"][k] = safe_proxy(v) if k != "no" else "present"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
            result["windows_proxy_enabled"] = winreg.QueryValueEx(key, "ProxyEnable")[0]
    except OSError:
        result["windows_proxy_enabled"] = None
    for env in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR", "PIP_CONFIG_FILE", "PIP_PROXY"):
        result[env + "_set"] = bool(os.environ.get(env))
    for label, url in (
        ("setuptools_release", "https://pypi.org/pypi/setuptools/80.9.0/json"),
        ("pypi_index", "https://pypi.org/simple/setuptools/"),
        ("torch_index", "https://download.pytorch.org/whl/cu128/torch/"),
    ):
        for mode in ("default_proxy_discovery", "direct_no_environment"):
            entry = {"target": label, "url": url, "mode": mode}
            with requests.Session() as session:
                session.trust_env = mode == "default_proxy_discovery"
                try:
                    # Public package metadata only. TLS verification remains enabled.
                    r = session.get(url, timeout=(6, 10))
                    entry["http_status"] = r.status_code
                    r.raise_for_status()
                    if label == "setuptools_release":
                        entry["version"] = r.json()["info"]["version"]
                    entry["ok"] = True
                except Exception as exc:
                    entry["ok"] = False
                    entry["error_type"] = type(exc).__name__
                    # Error text can include proxy credentials: store only its class.
            result["requests"].append(entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = root / "logs" / ("step01_network_" + stamp + ".json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}, ensure_ascii=False, indent=2))
    print("REPORT=" + str(output))


if __name__ == "__main__":
    main()
