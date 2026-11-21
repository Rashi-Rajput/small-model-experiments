import json
import unittest
from unittest.mock import Mock

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from task1_lora.data import HERE, make_dataset
from task1_lora.train import grade, install_adapters, training_pair


class InventoryTests(unittest.TestCase):
    def test_test_records_and_templates_are_held_out(self):
        data = json.loads((HERE / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(data, make_dataset())
        for field in ("template", "request", "target"):
            train = {json.dumps(row[field], sort_keys=True) for row in data["train"]}
            test = {json.dumps(row[field], sort_keys=True) for row in data["test"]}
            self.assertTrue(train.isdisjoint(test))

    def test_generated_json_requires_the_whole_typed_record(self):
        target = {"action": "issue", "sku": "NT-12", "quantity": 1}
        raw = '{"quantity":1, "sku":"NT-12", "action":"issue"}'
        self.assertTrue(grade(raw, target)["correct"])
        self.assertEqual(grade(raw.replace("NT-12", "NT-13"), target), {"schema": True, "correct": False})
        for bad in ("Here you go: " + raw, raw.replace(":1,", ":true,"),
                    raw.replace(":1,", ':"1",'), raw.replace(":1,", ':9,"quantity":1,')):
            self.assertFalse(grade(bad, target)["schema"])

    def test_only_response_and_end_token_have_targets(self):
        tokenizer = Mock(eos_token_id=2)
        tokenizer.apply_chat_template.return_value = [4, 5, 6]
        tokenizer.encode.return_value = [8, 9]
        inputs, labels = training_pair(tokenizer, dict(request="Count the stock.", target={"quantity": 1}))
        self.assertEqual(inputs.tolist(), [4, 5, 6, 8, 9, 2])
        self.assertEqual(labels.tolist(), [-100, -100, -100, 8, 9, 2])

    def test_adapters_start_at_the_base_and_receive_gradients(self):
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1)
        model = LlamaForCausalLM(config).eval()
        ids = torch.tensor([[3, 4, 5, 7]])
        with torch.no_grad():
            base = model(ids).logits
        weights = install_adapters(model)
        output = model(ids, labels=ids)
        torch.testing.assert_close(output.logits, base)
        output.loss.backward()
        self.assertEqual(len(weights), 4)
        self.assertTrue(all(param.grad is None for param in model.parameters() if not param.requires_grad))
        self.assertTrue(all(weights[name].grad.abs().sum() > 0 for name in weights if name.endswith("up.weight")))


if __name__ == "__main__":
    unittest.main()
