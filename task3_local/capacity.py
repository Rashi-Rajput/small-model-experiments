import argparse
import json
import re
import subprocess
from pathlib import Path

from local_model import MODEL, model_file, server_binary


def main():
    parser = argparse.ArgumentParser()
    server = Path(server_binary())
    parser.add_argument("--cli", default=server.with_name("llama-cli" + server.suffix))
    args = parser.parse_args()
    model = model_file("Q4_K_M")
    context = 1024
    command = [str(args.cli), "-m", str(model), "-ngl", "0", "-c", str(context),
               "-t", "4", "-n", "1", "-p", "Hi", "-st", "-lv", "5"]
    run = subprocess.run(command, check=True, capture_output=True, encoding="utf-8", errors="replace")
    log = run.stdout + run.stderr
    fields = {}
    for name in ("n_layer", "n_head_kv", "n_embd_head_k", "n_embd_head_v"):
        fields[name] = int(re.search(rf"print_info: {name}\s*= (\d+)", log)[1])
    cache_lines = [line.strip() for line in log.splitlines() if "CPU KV buffer size" in line]
    observed = float(re.search(r"=\s*([\d.]+) MiB", cache_lines[-1])[1])
    stride = fields["n_layer"] * fields["n_head_kv"] * (fields["n_embd_head_k"] + fields["n_embd_head_v"]) * 2
    if "K (f16)" not in log or "V (f16)" not in log or abs(observed - stride * context / 2**20) > 0.01:
        raise ValueError("Measured KV cache does not match the FP16 estimate")
    users, prompt, output = 20, 1792, 256
    cache = users * (prompt + output) * stride
    weights, reserve = model.stat().st_size, 4_000_000_000
    result = dict(model=MODEL + "-Q4_K_M", layers=fields["n_layer"], kv_heads=fields["n_head_kv"],
                  key_width=fields["n_embd_head_k"], value_width=fields["n_embd_head_v"], cache_dtype="f16",
                  bytes_per_token=stride, observed_context=context, observed_cache_mib=observed,
                  users=users, context_per_user=prompt + output, prompt_cap=prompt, generation_cap=output,
                  cache_bytes=cache, file_bytes=weights, assumed_workspace_bytes=reserve,
                  assumed_gpu_bytes=24_000_000_000, estimated_total_gb=(weights + cache + reserve) / 1e9)
    Path(__file__).with_name("capacity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
