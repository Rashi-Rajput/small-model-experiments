import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task1_lora.train import HERE, MODEL, REVISION, SETTINGS, generate, grade, install_adapters, load_adapter


CASES = [
    ("receive", "FP-39", 3, 14, "The delivery note said {old} of {sku}. We received {actual}."),
    ("receive", "DV-61", 9, 23, "Log the {actual} units of {sku} that arrived. The forecast was {old}."),
    ("issue", "LN-50", 4, 19, "The plan was to ship {old} of {sku}. The customer took {actual}."),
    ("issue", "MG-88", 7, 12, "We issued {actual} units of {sku}. The request was for {old}."),
    ("count", "AZ-41", 2, 17, "Our ledger lists {old} of {sku}. The physical count is {actual}."),
    ("count", "JC-76", 5, 21, "There are {actual} units of {sku} in the bin. The system still shows {old}."),
]


def main():
    torch.set_num_threads(SETTINGS["cpu_threads"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
    load_adapter(install_adapters(model))
    probes = []
    for action, sku, small, large, note in CASES:
        for actual, old in ((small, large), (large, small)):
            probes.append(dict(request=note.format(sku=sku, actual=actual, old=old),
                               target=dict(action=action, sku=sku, quantity=actual), other=old))
    outputs = generate(model, tokenizer, [p["request"] for p in probes])
    for row, out in zip(probes, outputs, strict=True):
        row["tuned"] = {**out, **grade(out["text"], row["target"])}
    passing_pairs = sum(a["tuned"]["correct"] and b["tuned"]["correct"]
                        for a, b in zip(probes[::2], probes[1::2], strict=True))
    data = dict(pairs=len(CASES), both_correct=passing_pairs, predictions=probes)
    (HERE / "quantity_check.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Both orders correct in {passing_pairs}/{len(CASES)} pairs.")


if __name__ == "__main__":
    main()
