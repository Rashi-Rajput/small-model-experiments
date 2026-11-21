import json
from pathlib import Path

import torch


def main():
    parameters = 7_000_000_000
    layers, width, rank = 32, 4096, 8
    adapter = layers * 2 * (width + width) * rank
    half = torch.empty(0, dtype=torch.bfloat16).element_size()
    single = torch.empty(0, dtype=torch.float32).element_size()
    trained = 2 * half + 3 * single
    reserve = 4_000_000_000
    block_size = 64
    states = {
        "full": parameters * trained,
        "lora": parameters * half + adapter * trained,
        "qlora": parameters * (0.5 + half / block_size) + adapter * trained,
    }
    result = dict(parameters=parameters, layers=layers, width=width, rank=rank, adapter=adapter,
                  bf16_bytes=half, fp32_bytes=single, trained_bytes_per_parameter=trained,
                  quant_block=block_size, assumed_temporary_bytes=reserve, model_state_bytes=states,
                  estimated_total_gb={k: (v + reserve) / 1e9 for k, v in states.items()})
    path = Path(__file__).with_name("memory.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["estimated_total_gb"])


if __name__ == "__main__":
    main()
