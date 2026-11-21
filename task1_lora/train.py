import csv
import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from encoder import Encoder, MODEL, REVISION


HERE = Path(__file__).resolve().parent
DATA = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data"
LABELS = {
    "card_arrival": "My ordered card has not arrived.",
    "card_delivery_estimate": "How long does delivery of a card usually take?",
    "cash_withdrawal_charge": "There is a fee for withdrawing cash from an ATM.",
    "wrong_amount_of_cash_received": "The cash machine gave me the wrong amount of money.",
    "declined_cash_withdrawal": "The cash machine refused my withdrawal.",
    "pending_cash_withdrawal": "My ATM withdrawal is still pending.",
    "pending_transfer": "A bank transfer is still pending.",
    "failed_transfer": "My transfer of money failed.",
    "transfer_not_received_by_recipient": "My recipient has not received the transfer.",
    "declined_card_payment": "A merchant declined my card payment.",
    "pending_card_payment": "A payment I made with my card is pending.",
    "card_payment_fee_charged": "I was charged a fee for paying by card.",
}
CONFIG = dict(rank=8, alpha=16, targets=["query", "value"], lr=0.0005, epochs=4, batch=16, seed=41)


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


def sample_split(name, per_label, excluded):
    with urlopen(f"{DATA}/{name}.csv", timeout=60) as response:
        rows = csv.DictReader(io.StringIO(response.read().decode("utf-8-sig")))
        groups = defaultdict(list)
        for row in rows:
            text, label = row["text"].strip(), row["category"]
            key = " ".join(text.casefold().split())
            if label in LABELS and key not in excluded:
                groups[label].append(text)
                excluded.add(key)
    rng = random.Random(CONFIG["seed"])
    return [(t, i) for i, label in enumerate(LABELS) for t in rng.sample(groups[label], per_label)]


@torch.no_grad()
def predict(model, rows, anchors):
    model.eval()
    predictions = []
    for start in range(0, len(rows), CONFIG["batch"]):
        texts = [t for t, _ in rows[start:start + CONFIG["batch"]]]
        predictions.extend((model(texts) @ anchors.T).argmax(dim=1).tolist())
    return predictions


def main():
    torch.manual_seed(CONFIG["seed"])
    torch.set_num_threads(3)
    rng = random.Random(CONFIG["seed"])
    seen = set()
    train = sample_split("train", 40, seen)
    test = sample_split("test", 16, seen)
    model = Encoder().eval()
    model.requires_grad_(False)
    base_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        anchors = model(list(LABELS.values()))
    baseline = predict(model, test, anchors)
    for name, layer in list(model.backbone.named_modules()):
        if name.split(".")[-1] in CONFIG["targets"] and isinstance(layer, nn.Linear):
            model.backbone.set_submodule(name, LoRALinear(layer, CONFIG["rank"], CONFIG["alpha"]))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and predict(model, test, anchors) == baseline
    optimizer = torch.optim.AdamW(trainable.values(), lr=CONFIG["lr"])
    losses = []
    model.train()
    for epoch in range(CONFIG["epochs"]):
        rng.shuffle(train)
        total = 0
        for i in range(0, len(train), CONFIG["batch"]):
            batch = train[i:i + CONFIG["batch"]]
            scores = 20 * model([t for t, _ in batch]) @ anchors.T
            loss = F.cross_entropy(scores, torch.tensor([y for _, y in batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(train))
        print(f"Epoch {epoch + 1}: {losses[-1]:.4f}", flush=True)
    tuned = predict(model, test, anchors)
    labels = list(LABELS)
    records = [dict(text=t, gold=labels[y], base=labels[b], tuned=labels[f])
               for (t, y), b, f in zip(test, baseline, tuned, strict=True)]
    result = dict(model=MODEL, revision=REVISION, settings=CONFIG, train_size=len(train),
                  base_parameters=base_params, adapter_parameters=sum(p.numel() for p in trainable.values()),
                  train_losses=losses, test_size=len(test), predictions=records)
    for key in ("base", "tuned"):
        result[key + "_correct"] = sum(r[key] == r["gold"] for r in records)
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save_file({k: v.detach().contiguous() for k, v in trainable.items()}, HERE / "adapter.safetensors")
    print(result["base_correct"], result["tuned_correct"], "/", len(test))


if __name__ == "__main__":
    main()
