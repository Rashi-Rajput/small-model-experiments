import hashlib
import shutil
from urllib.request import urlopen

from local_model import CACHE, model_file


SOURCE = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9"
HASHES = {
    "Q4_K_M": "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
    "Q8_0": "d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    CACHE.mkdir(exist_ok=True)
    for quant, expected in HASHES.items():
        path = model_file(quant)
        if not path.exists() or digest(path) != expected:
            partial = path.with_suffix(".part")
            with urlopen(f"{SOURCE}/{path.name}", timeout=120) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            if digest(partial) != expected:
                raise ValueError(f"Checksum mismatch for {path.name}")
            partial.replace(path)
        print(path.name, expected)


if __name__ == "__main__":
    main()
