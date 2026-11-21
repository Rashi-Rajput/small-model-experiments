import json
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


CACHE = Path(__file__).resolve().parent / ".cache"


def chat(url, messages, limit, grammar=None):
    payload = dict(messages=messages, temperature=0, seed=41, max_tokens=limit,
                   cache_prompt=False, timings_per_token=False)
    if grammar is not None:
        payload["grammar"] = grammar
    request = Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


@contextmanager
def serve(binary, model, context):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
               "-ngl", "0", "-t", "4", "-tb", "4", "-np", "1", "-c", str(context)]
    CACHE.mkdir(exist_ok=True)
    with (CACHE / f"server-{port}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited. See {log.name}")
                try:
                    with urlopen(url + "/health", timeout=2):
                        break
                except (URLError, TimeoutError):
                    time.sleep(0.2)
            else:
                raise TimeoutError(f"Server did not start. See {log.name}")
            yield url, process
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def server_binary():
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    return os.environ.get("LLAMA_SERVER", str(CACHE / "llama" / name))


def model_file(quant):
    return CACHE / f"SmolLM2-135M-Instruct-{quant}.gguf"
