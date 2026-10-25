"""
Start everything the demo needs, then open the browser.

    python3 launch.py            (or: make demo)
    python3 launch.py --port 8421 --no-browser
    python3 launch.py --reload   (restart the server on code changes)

Steps, each skipped if already done/running:
    .venv -> .env -> Postgres (Docker) -> Ollama + models -> load data (first run) -> web UI

Ctrl+C stops the web server and any Ollama this script started.
Postgres keeps running; `make db-down` stops it.
"""

import argparse
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "bin" / "python"
EMBED_MODEL = "nomic-embed-text"

# processes we started, so we can stop them on exit
_children: list[subprocess.Popen] = []


# ------------------------------------------------------------------ output helpers


def ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}", flush=True)


def doing(msg: str) -> None:
    print(f"  \033[36m→\033[0m {msg}", flush=True)


def fail(msg: str, hint: str = "") -> None:
    print(f"\n  \033[31m✗ {msg}\033[0m", file=sys.stderr)
    if hint:
        print(f"    {hint}", file=sys.stderr)
    shutdown()
    sys.exit(1)


def run(cmd: list[str], **kwargs) -> None:
    """Run a setup command, streaming its output, and stop on failure."""
    result = subprocess.run(cmd, cwd=ROOT, text=True, **kwargs)
    if result.returncode != 0:
        # quiet steps hid their output; show it now
        for captured in (result.stdout, result.stderr):
            if captured:
                print(captured, file=sys.stderr)
        fail(f"Command failed: {' '.join(cmd)}")


def wait_for(check, timeout: float, interval: float = 0.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if check():
            return True
        time.sleep(interval)
    return False


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


# ------------------------------------------------------------------ virtualenv


def ensure_venv() -> None:
    """Create .venv on first run, then re-run this script inside it."""
    if pathlib.Path(sys.prefix).resolve() == VENV.resolve():
        return
    if not VENV_PYTHON.exists():
        doing("Creating .venv and installing dependencies (first run only)...")
        run([sys.executable, "-m", "venv", str(VENV)])
        run([str(VENV_PYTHON), "-m", "pip", "install", "-q", "--upgrade", "pip"])
        run([str(VENV_PYTHON), "-m", "pip", "install", "-q", "-e", ".[dev]"])
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(ROOT / "launch.py"), *sys.argv[1:]])


# ------------------------------------------------------------------ .env


def ensure_env_file() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        shutil.copy(ROOT / ".env.example", env_file)
        ok("Created .env from .env.example")


# ------------------------------------------------------------------ postgres


def _dsn_parts(dsn: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in dsn.split() if "=" in part)


def ensure_postgres(dsn: str) -> None:
    import psycopg2

    def can_connect() -> bool:
        try:
            psycopg2.connect(dsn, connect_timeout=2).close()
            return True
        except psycopg2.OperationalError:
            return False

    if can_connect():
        ok("Postgres is running")
        return

    # only touch Docker if DB_DSN points at our container; any other database is yours to start
    parts = _dsn_parts(dsn)
    compose_port = os.getenv("POSTGRES_PORT", "5433")
    if parts.get("host") not in ("localhost", "127.0.0.1") or parts.get("port") != compose_port:
        fail(
            f"Can't connect to the database in DB_DSN ({parts.get('host')}:{parts.get('port')}).",
            "Start that Postgres, or point DB_DSN in .env at the Docker one (see .env.example).",
        )

    if not shutil.which("docker"):
        fail("Docker isn't installed.", "Install Docker Desktop: https://www.docker.com/products/docker-desktop/")

    def docker_up() -> bool:
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0

    if not docker_up():
        if sys.platform == "darwin":
            doing("Starting Docker Desktop...")
            subprocess.run(["open", "-a", "Docker"], capture_output=True)
            if not wait_for(docker_up, timeout=90, interval=2):
                fail("Docker Desktop didn't start within 90s.", "Open it manually, then rerun.")
        else:
            fail("Docker isn't running.", "Start the Docker daemon, then rerun.")

    doing("Starting Postgres in Docker...")
    run(["docker", "compose", "up", "-d", "--wait", "db"], capture_output=True)
    if not wait_for(can_connect, timeout=30):
        fail("Postgres container started but isn't accepting connections.", "Check: docker compose logs db")
    ok("Postgres is running (Docker, port %s)" % compose_port)


# ------------------------------------------------------------------ ollama


def ensure_ollama(base_url: str, chat_model: str | None) -> None:
    api = base_url.removesuffix("/v1").rstrip("/")

    if http_ok(f"{api}/api/version"):
        ok("Ollama is running")
    else:
        if not shutil.which("ollama"):
            fail("Ollama isn't installed.", "Install it from https://ollama.com, then rerun.")
        doing("Starting Ollama...")
        log = open(ROOT / ".ollama.log", "w")
        _children.append(
            subprocess.Popen(["ollama", "serve"], stdout=log, stderr=subprocess.STDOUT)
        )
        if not wait_for(lambda: http_ok(f"{api}/api/version"), timeout=30):
            fail("Ollama didn't start within 30s.", "See .ollama.log for details.")
        ok("Ollama is running (started by launcher)")

    import json

    with urllib.request.urlopen(f"{api}/api/tags", timeout=5) as resp:
        installed = {m["name"] for m in json.load(resp)["models"]}

    wanted = [EMBED_MODEL] + ([chat_model] if chat_model else [])
    for model in wanted:
        if model in installed or f"{model}:latest" in installed:
            continue
        doing(f"Pulling {model} (one-time download)...")
        run(["ollama", "pull", model])
    ok(f"Models ready: {', '.join(wanted)}")


# ------------------------------------------------------------------ data


def _table_rows(dsn: str, table: str) -> int:
    import psycopg2

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        if cur.fetchone()[0] is None:
            return 0
        cur.execute(f"SELECT count(*) FROM {table}")
        return cur.fetchone()[0]


def ensure_data(dsn: str) -> None:
    planets = _table_rows(dsn, "planets")
    if planets == 0:
        if not (ROOT / "data" / "raw_pscomppars.csv").exists():
            doing("Downloading the NASA Exoplanet Archive table...")
            run([sys.executable, "data/download_data.py"])
        doing("Loading planets and host stars...")
        run([sys.executable, "data/load_to_postgres.py"], capture_output=True)
        planets = _table_rows(dsn, "planets")

    chunks = _table_rows(dsn, "wiki_chunks")
    if chunks == 0:
        doing("Embedding and loading the wiki corpus (about a minute)...")
        run([sys.executable, "data/load_wiki_to_postgres.py"], capture_output=True)
        chunks = _table_rows(dsn, "wiki_chunks")

    ok(f"Data loaded: {planets:,} planets, {chunks:,} wiki passages")


# ------------------------------------------------------------------ web server


def start_server(port: int, reload: bool, open_browser: bool) -> subprocess.Popen:
    if port_in_use(port):
        fail(f"Port {port} is already in use.", f"Stop whatever is using it, or pass --port {port + 1}.")

    cmd = [sys.executable, "-m", "uvicorn", "frontend.server:app", "--port", str(port)]
    if reload:
        cmd.append("--reload")
    doing("Starting the web UI...")
    server = subprocess.Popen(cmd, cwd=ROOT)
    _children.append(server)

    url = f"http://localhost:{port}"
    if not wait_for(lambda: http_ok(f"{url}/api/health") or server.poll() is not None, timeout=60):
        fail("The web server didn't come up within 60s.")
    if server.poll() is not None:
        fail("The web server exited during startup (see the error above).")

    print(f"\n  \033[1mexo-agent is running at {url}\033[0m   (Ctrl+C to stop)\n", flush=True)
    if open_browser:
        webbrowser.open(url)
    return server


def shutdown() -> None:
    for proc in reversed(_children):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    _children.clear()


# ------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description="Start everything exo-agent needs.")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    print("\n  exo-agent launcher\n")
    ensure_venv()
    ensure_env_file()

    # import late: needs the venv's packages and the .env we may have just created
    sys.path.insert(0, str(ROOT / "src"))
    from exo_agent import config

    ensure_postgres(config.DB_DSN)
    ensure_ollama(config.OLLAMA_BASE_URL, config.MODEL if config.PROVIDER == "ollama" else None)
    ensure_data(config.DB_DSN)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    server = start_server(args.port, args.reload, not args.no_browser)
    try:
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  Shutting down... (Postgres stays up; `make db-down` stops it)")
        shutdown()


if __name__ == "__main__":
    main()
