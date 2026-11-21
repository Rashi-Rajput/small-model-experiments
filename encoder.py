from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "vocab.txt", "special_tokens_map.json"]
        for filename in files:
            local = hf_hub_download(MODEL, filename, revision=REVISION)
        path = Path(local).parent
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.backbone = AutoModel.from_pretrained(path)

    def forward(self, texts):
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        states = self.backbone(**batch).last_hidden_state
        mask = batch.attention_mask.unsqueeze(-1)
        return normalize((states * mask).sum(dim=1) / mask.sum(dim=1), dim=1)
