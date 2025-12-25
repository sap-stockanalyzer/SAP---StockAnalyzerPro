import os
import sys
import time
import signal
import warnings
import subprocess
import platform
import threading
from typing import Optional
from dotenv import load_dotenv

from utils.live_log import append_log, prune_old_logs

load_dotenv()

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


def launch(cmd, name: str) -> subprocess.Popen:
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
    # NOTE: proc.stdout can be None if Popen wasn't created with stdout=PIPE.
    if proc.stdout is None:
        return
    for line in proc.stdout:
        # Mirror to terminal
        try:
            print(line, end="")
        except Exception:
            pass
        # Persist to live log
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

    print("────────────────────────────────────────")
    print("🚀 Launching AION Analytics system")
    print(f"   • backend          → http://{backend_host}:{backend_port}")
    print(f"   • dt_backend       → http://{dt_host}:{dt_port}")
    print(f"   • replay_service   → http://{replay_host}:{replay_port} (dormant)")
    print("────────────────────────────────────────", flush=True)

    append_log("AION system starting")
    prune_old_logs()

    backend_proc = launch(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.backend_service:app",
            "--host",
            backend_host,
            "--port",
            str(backend_port),
        ],
        "backend",
    )
    threading.Thread(target=pipe_output, args=(backend_proc, "backend"), daemon=True).start()

    dt_proc = launch(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "dt_backend.fastapi_main:app",
            "--host",
            dt_host,
            "--port",
            str(dt_port),
        ],
        "dt_backend",
    )
    threading.Thread(target=pipe_output, args=(dt_proc, "dt_backend"), daemon=True).start()

    dt_live_proc = launch(
        [
            sys.executable,
            "-m",
            "dt_backend.jobs.live_market_data_loop",
        ],
        "dt_backend_live_loop",
    )
    threading.Thread(target=pipe_output, args=(dt_live_proc, "dt_backend_live_loop"), daemon=True).start()

    replay_proc = launch(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "replay_service:app",
            "--host",
            replay_host,
            "--port",
            str(replay_port),
        ],
        "replay_service",
    )
    threading.Thread(target=pipe_output, args=(replay_proc, "replay_service"), daemon=True).start()

    try:
        while True:
            time.sleep(1)

            if backend_proc.poll() is not None:
                raise RuntimeError("backend process exited")

            if dt_proc.poll() is not None:
                raise RuntimeError("dt_backend API process exited")

            if dt_live_proc.poll() is not None:
                raise RuntimeError("dt_backend live loop exited")

            if replay_proc.poll() is not None:
                raise RuntimeError("replay service exited")

    except KeyboardInterrupt:
        print("\n🛑 Shutting down AION system...")
        append_log("AION system shutdown requested (KeyboardInterrupt)")

    finally:
        shutdown_process(replay_proc, "replay_service")
        shutdown_process(dt_live_proc, "dt_backend_live_loop")
        shutdown_process(dt_proc, "dt_backend")
        shutdown_process(backend_proc, "backend")
        time.sleep(2)
        append_log("AION system stopped")
