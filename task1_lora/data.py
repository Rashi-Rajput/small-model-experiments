import json
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
SKUS = ["BK-14", "WT-20", "PN-07", "HD-32", "MT-15", "GL-08", "SP-19", "NT-12",
        "BX-25", "RB-16", "PL-03", "CL-24"]
NEW_SKUS = ["TK-31", "BL-06", "CP-47", "HZ-11", "QD-29", "VR-53"]
WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two "
         "twenty-three twenty-four").split()
TRAIN = {
    "receive": ["Receive {n} units of {sku}.", "Put {n} of {sku} into stock.",
                "We took delivery of {n} units of {sku}.", "Incoming shipment: {sku}, quantity {n}.",
                "Add the {n} new {sku} units to the shelf.", "{sku} arrived today. Book {n}.",
                "The supplier delivered {n} of {sku}, although we ordered {other}.",
                "We expected {other} units of {sku}, but only {n} arrived."],
    "issue": ["Issue {n} units of {sku}.", "Send {n} of {sku} to the workshop.",
              "We shipped {n} units of {sku}.", "Outgoing order: {sku}, quantity {n}.",
              "Take {n} of {sku} out of stock.", "The technician picked up {n} of {sku}.",
              "The customer ordered {other} of {sku}. We dispatched {n}.",
              "Record the {n} units of {sku} that left, not the {other} requested."],
    "count": ["The shelf count for {sku} is {n}.", "We counted {n} units of {sku}.",
              "Stocktake: {sku}, quantity {n}.", "Set the on-hand total for {sku} to {n}.",
              "An audit found {n} of {sku} on hand.", "{sku} has a verified balance of {n}.",
              "The ledger says {other} for {sku}, but the shelf holds {n}.",
              "Our actual count of {sku} is {n}; the old figure was {other}."],
}
TEST = {
    "receive": ["Can you log the delivery of {n} pieces of {sku}?",
                "{sku} came in on the morning truck, {n} in all.",
                "The packing slip lists {other} of {sku}. We accepted {n} at the dock.",
                "Of the {other} units of {sku} due in, the warehouse signed for {n}."],
    "issue": ["Book {n} of {sku} against the batch that left today.",
              "A courier collected {n} pieces of {sku} from our shelf.",
              "The pick list called for {other} of {sku}. The number sent was {n}.",
              "For {sku}, dispatch confirmed {n} pieces rather than the proposed {other}."],
    "count": ["During the recount we found {n} pieces of {sku}.",
              "At closing, the bin for {sku} contained {n} units.",
              "The system shows {other} of {sku}. A physical check puts it at {n}.",
              "Replace yesterday's {other} on the {sku} count sheet with the verified {n}."],
}


def make_dataset():
    rng = random.Random(2718)
    used = set()
    splits = {}
    for split, templates, per_template in [("train", TRAIN, 8), ("test", TEST, 6)]:
        rows = []
        for action, forms in templates.items():
            for form_id, form in enumerate(forms, 1):
                for i in range(per_template):
                    pool = NEW_SKUS if split == "test" and i % 2 else SKUS
                    while True:
                        sku, quantity = rng.choice(pool), rng.randint(1, 24)
                        key = (action, sku, quantity)
                        if key not in used:
                            used.add(key)
                            break
                    other = rng.randint(quantity + 1, quantity + 12)
                    number = WORDS[quantity] if i % 3 == 0 else str(quantity)
                    rows.append(dict(template=f"{split}-{action}-{form_id}",
                                     request=form.format(sku=sku, n=number, other=other),
                                     target=dict(action=action, sku=sku, quantity=quantity)))
        rng.shuffle(rows)
        splits[split] = rows
    return splits


if __name__ == "__main__":
    (HERE / "dataset.json").write_text(json.dumps(make_dataset(), indent=2) + "\n", encoding="utf-8")
