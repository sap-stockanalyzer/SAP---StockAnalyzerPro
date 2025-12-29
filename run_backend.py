import os
import sys
import time
import signal
import warnings
import subprocess
import platform
import threading
from typing import Optional



import importlib.util

def module_exists(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False

# dotenv is optional; if .env has bad lines we don't want to crash boot
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from utils.live_log import append_log, prune_old_logs

from backend.core.config import PATHS
print("ROLLING:", PATHS.get("rolling"))
print("ROLLING_BODY:", PATHS.get("rolling_body"))
print("ROLLING_BRAIN:", PATHS.get("rolling_brain"))

# Quiet noisy libs
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module=r"pandas_ta\.__init__",
)
os.environ.setdefault("PYTHONWARNINGS", "ignore:::pandas_ta.__init__")


def _is_valid_bind_host(host: str) -> bool:
    h = (host or "").strip().lower()
    if not h:
        return False
    # "0.0.0.1" is NOT a valid bind address; treat as invalid.
    if h == "0.0.0.1":
        return False
    return True


def _normalize_bind_host(host: str, default: str = "0.0.0.0") -> str:
    return host if _is_valid_bind_host(host) else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name, "") or "").strip().lower()
    if v == "":
        return default
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def uvicorn_cmd(
    app: str,
    host: str,
    port: str | int,
    *,
    log_level: str = "warning",
    access_log: bool = False,
    reload: bool = False,
) -> list[str]:
    """
    Build a safe uvicorn command.
    IMPORTANT: flags are presence-based. Never pass "false"/"true" as extra args.
    """
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        app,
        "--host",
        str(host),
        "--port",
        str(port),
        "--log-level",
        str(log_level),
    ]

    # access log is a boolean flag in uvicorn; use --no-access-log to disable.
    if not access_log:
        cmd.append("--no-access-log")

    if reload:
        cmd.append("--reload")

    return cmd


def launch(cmd: list[str], name: str) -> subprocess.Popen:
    print(f"🚀 Starting {name}...")
    append_log(f"Starting {name}")
    prune_old_logs()

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
        env=os.environ.copy(),
    )


def pipe_output(proc: subprocess.Popen, name: str):
    if proc.stdout is None:
        return
    for line in proc.stdout:
        try:
            print(line, end="")
        except Exception:
            pass
        append_log(f"[{name}] {line}")


def shutdown_process(p: Optional[subprocess.Popen], name: str = ""):
    if not p or p.poll() is not None:
        return

    try:
        append_log(f"Stopping {name or 'process'} (pid={p.pid})")
    except Exception:
        pass

    try:
        if platform.system() == "Windows":
            p.terminate()
        else:
            # SIGINT lets uvicorn shutdown gracefully
            p.send_signal(signal.SIGINT)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


if __name__ == "__main__":
    # Bind hosts (server should be 0.0.0.0 to listen externally)
    backend_host = _normalize_bind_host(os.environ.get("APP_HOST", "0.0.0.0"))
    backend_port = os.environ.get("APP_PORT", "8000")

    dt_host = _normalize_bind_host(os.environ.get("DT_APP_HOST", "0.0.0.0"))
    dt_port = os.environ.get("DT_APP_PORT", "8010")

    replay_host = _normalize_bind_host(os.environ.get("REPLAY_APP_HOST", "0.0.0.0"))
    replay_port = os.environ.get("REPLAY_APP_PORT", "8020")

    # Optional toggles
    UVICORN_LOG_LEVEL = os.environ.get("UVICORN_LOG_LEVEL", "warning").strip() or "warning"
    UVICORN_ACCESS_LOG = _env_bool("UVICORN_ACCESS_LOG", default=False)
    UVICORN_RELOAD = _env_bool("UVICORN_RELOAD", default=False)

    print("────────────────────────────────────────")
    print("🚀 Launching AION Analytics system")
    print(f"   • backend          → http://{backend_host}:{backend_port}")
    print(f"   • dt_backend       → http://{dt_host}:{dt_port}")
    print(f"   • replay_service   → http://{replay_host}:{replay_port} (dormant)")
    print("────────────────────────────────────────", flush=True)

    append_log("AION system starting")
    prune_old_logs()

    backend_proc = launch(
        uvicorn_cmd(
            "backend.backend_service:app",
            backend_host,
            backend_port,
            log_level=UVICORN_LOG_LEVEL,
            access_log=UVICORN_ACCESS_LOG,
            reload=UVICORN_RELOAD,
        ),
        "backend",
    )
    threading.Thread(target=pipe_output, args=(backend_proc, "backend"), daemon=True).start()

    dt_proc = launch(
        uvicorn_cmd(
            "dt_backend.fastapi_main:app",
            dt_host,
            dt_port,
            log_level=UVICORN_LOG_LEVEL,
            access_log=UVICORN_ACCESS_LOG,
            reload=UVICORN_RELOAD,
        ),
        "dt_backend",
    )
    threading.Thread(target=pipe_output, args=(dt_proc, "dt_backend"), daemon=True).start()
    # DT worker: prefer dt_scheduler if present; otherwise fall back to the legacy live loop.
    dt_worker_module = (
        "dt_backend.jobs.dt_scheduler"
        if module_exists("dt_backend.jobs.dt_scheduler")
        else "dt_backend.jobs.live_market_data_loop"
    )
    dt_worker_name = "dt_scheduler" if dt_worker_module.endswith("dt_scheduler") else "dt_backend_live_loop"
    dt_worker_proc = launch([sys.executable, "-m", dt_worker_module], dt_worker_name)
    threading.Thread(
        target=pipe_output,
        args=(dt_worker_proc, dt_worker_name),
        daemon=True,
    ).start()

    replay_proc = launch(
        uvicorn_cmd(
            "replay_service:app",
            replay_host,
            replay_port,
            log_level=UVICORN_LOG_LEVEL,
            access_log=UVICORN_ACCESS_LOG,
            reload=UVICORN_RELOAD,
        ),
        "replay_service",
    )
    threading.Thread(target=pipe_output, args=(replay_proc, "replay_service"), daemon=True).start()

    try:
        while True:
            if backend_proc.poll() is not None:
                raise RuntimeError("backend process exited")

            if dt_proc.poll() is not None:
                raise RuntimeError("dt_backend (API) process exited")

            # DT worker is optional: if it dies, restart it instead of killing the whole stack.
            if dt_worker_proc.poll() is not None:
                rc = dt_worker_proc.returncode
                msg = f"[supervisor] ⚠️ {dt_worker_name} exited (code={rc}) — restarting…"
                print(msg, flush=True)
                append_log(msg)
                time.sleep(3)
                dt_worker_proc = launch([sys.executable, "-m", dt_worker_module], dt_worker_name)
                threading.Thread(
                    target=pipe_output, args=(dt_worker_proc, dt_worker_name), daemon=True
                ).start()

            # replay_service is dormant/optional
            if replay_proc is not None and replay_proc.poll() is not None:
                rc = replay_proc.returncode
                msg = f"[supervisor] ⚠️ replay_service exited (code={rc}) — continuing without it"
                print(msg, flush=True)
                append_log(msg)
                replay_proc = None

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down AION system...")
        append_log("AION system shutdown requested (KeyboardInterrupt)")

    finally:
        shutdown_process(replay_proc, "replay_service")
        shutdown_process(dt_worker_proc, dt_worker_name)
        shutdown_process(dt_proc, "dt_backend")
        shutdown_process(backend_proc, "backend")
        time.sleep(2)
        append_log("AION system stopped")
