import argparse
import json
import math
import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SYSTEM = ('Read the inventory note. Return only JSON with keys "action", "sku", and "quantity". '
          'Use "receive" for arrivals, "issue" for departures, or "count" for stocktakes. '
          'Copy the SKU and use the confirmed quantity as an integer.')
SETTINGS = dict(rank=8, alpha=16, targets=["q_proj", "v_proj"], lr=0.0008, epochs=3,
                batch=8, max_new_tokens=48, seed=41, cpu_threads=4)


class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.down = nn.Linear(linear.in_features, rank, bias=False)
        self.up = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.linear(x) + self.up(self.down(x)) * self.scale


def install_adapters(model):
    model.requires_grad_(False)
    for layer in model.model.layers:
        for name in SETTINGS["targets"]:
            linear = getattr(layer.self_attn, name)
            setattr(layer.self_attn, name, LoRALinear(linear, SETTINGS["rank"], SETTINGS["alpha"]))
    return {k: p for k, p in model.named_parameters() if p.requires_grad}


def prompt_ids(tokenizer, note):
    messages = [dict(role="system", content=SYSTEM), dict(role="user", content=note)]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)


def training_pair(tokenizer, row):
    prompt = prompt_ids(tokenizer, row["request"])
    text = json.dumps(row["target"], separators=(",", ":"))
    target = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    return torch.tensor(prompt + target), torch.tensor([-100] * len(prompt) + target)


def train(model, tokenizer, rows, weights):
    pairs = [training_pair(tokenizer, row) for row in rows]
    optimizer = torch.optim.AdamW(weights.values(), lr=SETTINGS["lr"])
    rng = random.Random(SETTINGS["seed"])
    model.train()
    losses = []
    for epoch in range(SETTINGS["epochs"]):
        rng.shuffle(pairs)
        total, count = 0.0, 0
        for start in range(0, len(pairs), SETTINGS["batch"]):
            batch = pairs[start:start + SETTINGS["batch"]]
            inputs = pad_sequence([x for x, _ in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
            labels = pad_sequence([y for _, y in batch], batch_first=True, padding_value=-100)
            mask = torch.arange(inputs.size(1))[None, :] < torch.tensor([len(x) for x, _ in batch])[:, None]
            states = model.model(input_ids=inputs, attention_mask=mask, use_cache=False).last_hidden_state
            targets = labels[:, 1:]
            supervised = targets != -100
            logits = model.lm_head(states[:, :-1][supervised])
            loss = F.cross_entropy(logits, targets[supervised])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tokens = int(supervised.sum())
            total += loss.item() * tokens
            count += tokens
            if start % (SETTINGS["batch"] * 8) == 0:
                print(f"Epoch {epoch + 1}, batch {start // SETTINGS['batch'] + 1}: {loss.item():.3f}", flush=True)
        losses.append(total / count)
    return losses


@torch.inference_mode()
def generate(model, tokenizer, notes):
    model.eval()
    output = []
    for start in range(0, len(notes), SETTINGS["batch"]):
        prompts = [prompt_ids(tokenizer, s) for s in notes[start:start + SETTINGS["batch"]]]
        batch = tokenizer.pad({"input_ids": prompts}, padding=True, return_tensors="pt", padding_side="left")
        ids = model.generate(**batch, do_sample=False, max_new_tokens=SETTINGS["max_new_tokens"],
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        for row in ids[:, batch.input_ids.shape[1]:].tolist():
            ended = tokenizer.eos_token_id in row
            if ended:
                row = row[:row.index(tokenizer.eos_token_id)]
            output.append(dict(text=tokenizer.decode(row, skip_special_tokens=False), ended=ended))
    return output


def unique_keys(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def grade(raw, target):
    try:
        value = json.loads(raw, object_pairs_hook=unique_keys)
    except ValueError:
        return dict(schema=False, correct=False)
    schema = (isinstance(value, dict) and value.keys() == target.keys()
              and value["action"] in ("receive", "issue", "count")
              and type(value["sku"]) is str and type(value["quantity"]) is int)
    return dict(schema=schema, correct=schema and value == target)


def load_adapter(weights):
    saved = load_file(HERE / "adapter.safetensors")
    if saved.keys() != weights.keys():
        raise ValueError("Adapter tensors do not match the decoder")
    with torch.no_grad():
        for name, param in weights.items():
            param.copy_(saved[name])


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ask")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(SETTINGS["seed"])
    torch.set_num_threads(SETTINGS["cpu_threads"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
    if args.ask is not None:
        load_adapter(install_adapters(model))
        print(generate(model, tokenizer, [args.ask])[0]["text"])
        return
    splits = json.loads((HERE / "dataset.json").read_text(encoding="utf-8"))
    notes = [row["request"] for row in splits["test"]]
    if args.verify:
        load_adapter(install_adapters(model))
        saved = json.loads((HERE / "results.json").read_text(encoding="utf-8"))["predictions"]
        outputs = generate(model, tokenizer, notes)
        for actual, record in zip(outputs, saved, strict=True):
            expected = record["tuned"]
            if actual["text"] != expected["text"] or actual["ended"] != expected["ended"]:
                raise RuntimeError(f"Reloaded adapter disagrees on {record['request']!r}")
        print(f"Reloaded adapter reproduced all {len(saved)} continuations.")
        return
    base_parameters = sum(p.numel() for p in model.parameters())
    print("Generating with the base checkpoint", flush=True)
    baseline = generate(model, tokenizer, notes)
    weights = install_adapters(model)
    losses = train(model, tokenizer, splits["train"], weights)
    print("Generating with the trained adapter", flush=True)
    tuned = generate(model, tokenizer, notes)
    predictions = []
    for row, base, adapted in zip(splits["test"], baseline, tuned, strict=True):
        predictions.append(dict(**row, base={**base, **grade(base["text"], row["target"])},
                                tuned={**adapted, **grade(adapted["text"], row["target"])}))
    result = dict(model=MODEL, revision=REVISION, architecture=type(model).__name__, settings=SETTINGS,
                  system_prompt=SYSTEM, base_parameters=base_parameters, train_size=len(splits["train"]),
                  test_size=len(notes), adapter_parameters=sum(p.numel() for p in weights.values()),
                  train_losses=losses, predictions=predictions)
    for name in ("base", "tuned"):
        for metric in ("schema", "correct"):
            result[f"{name}_{metric}"] = sum(p[name][metric] for p in predictions)
    save_file({k: v.detach().contiguous() for k, v in weights.items()}, HERE / "adapter.safetensors",
              metadata=dict(base=MODEL, revision=REVISION))
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Exact records:", result["base_correct"], "base,", result["tuned_correct"], "tuned, of", len(notes))


if __name__ == "__main__":
    main()
