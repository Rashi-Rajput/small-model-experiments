import hashlib
import shutil
from urllib.request import urlopen

from local_model import CACHE, model_file


SOURCE = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c"
HASHES = {
    "Q4_K_M": "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d",
    "Q8_0": "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83",
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
