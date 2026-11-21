import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pymupdf
import torch

from encoder import Encoder
from local_model import CACHE, chat, model_file, serve, server_binary


HERE = Path(__file__).resolve().parent
SOURCE = "https://www.rfc-editor.org/rfc/rfc9110.pdf"
DIGEST = "60b30efa1048900833d1758440247fe8ac85a3134f2327388dcb24e07d814c89"
SIZES = (64, 192)
OVERLAP = 16


def read_pages():
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "rfc9110.pdf"
    if not path.exists():
        with urlopen(SOURCE, timeout=60) as response:
            path.write_bytes(response.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("PDF differs from the evaluated edition")
    with pymupdf.open(path) as doc:
        pages = []
        for page in doc:
            text = page.get_text(sort=True)
            text = re.sub(r"(?m)^\s*RFC 9110\s+HTTP Semantics\s+June 2022\s*$", "", text)
            text = re.sub(r"(?m)^\s*Fielding, et al\.\s+Standards Track\s+Page \d+\s*$", "", text)
            pages.append(" ".join(text.split()))
        return pages


@torch.inference_mode()
def embed(model, texts):
    return np.concatenate([model(texts[i:i + 24]).numpy() for i in range(0, len(texts), 24)])


def index_pages(model, pages, size):
    cache = CACHE / f"rfc9110-{DIGEST[:8]}-minilm-{size}-{OVERLAP}.npz"
    chunks = []
    for number, page in enumerate(pages, start=1):
        offsets = model.tokenizer(page, add_special_tokens=False, return_offsets_mapping=True,
                                  verbose=False)["offset_mapping"]
        for i in range(0, len(offsets), size - OVERLAP):
            last = min(i + size, len(offsets))
            start, end = offsets[i][0], offsets[last - 1][1]
            chunks.append(dict(page=number, start=start, end=end, text=page[start:end]))
            if last == len(offsets):
                break
    fingerprint = hashlib.sha256(json.dumps(chunks).encode()).hexdigest()
    if cache.exists():
        with np.load(cache) as saved:
            if saved["fingerprint"].item() == fingerprint:
                return chunks, saved["vectors"]
    vectors = embed(model, [c["text"] for c in chunks])
    np.savez(cache, vectors=vectors, fingerprint=fingerprint)
    return chunks, vectors


def search(chunks, vectors, query):
    scores = vectors @ query
    return [dict(**chunks[i], score=float(scores[i])) for i in np.argsort(-scores, kind="stable")[:5]]


def answer(url, model, question, vector, pages, hits):
    candidates = {}
    for hit in hits:
        page = pages[hit["page"] - 1]
        for sentence in re.finditer(r"\S.*?(?:[.!?](?=\s+[A-Z0-9•]|$)|$)", page):
            if hit["start"] <= sentence.start() and sentence.end() <= hit["end"]:
                text = sentence.group()
                if len(text.split()) >= 5:
                    candidates[(hit["page"], text)] = None
    if not candidates:
        return dict(answer=None, page=None, choices=[])
    passages = list(candidates)
    scores = embed(model, [s for _, s in passages]) @ vector
    choices = [passages[i] for i in np.argsort(-scores, kind="stable")[:4]]
    options = "\n".join(f"{i}. {s}" for i, (_, s) in enumerate(choices, 1))
    messages = [dict(role="system", content="Choose the sentence that answers the question. "
                     "Return its number, or 0 when no sentence answers it."),
                dict(role="user", content=f"{question}\n{options}")]
    grammar = f"root ::= [0-{len(choices)}]"
    selected = int(chat(url, messages, 3, grammar)["choices"][0]["message"]["content"])
    page, text = choices[selected - 1] if selected else (None, None)
    return dict(answer=text, page=page, choices=choices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask")
    parser.add_argument("--size", type=int, choices=SIZES, default=192)
    parser.add_argument("--answers", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    pages = read_pages()
    model = Encoder().eval()
    if args.ask:
        chunks, vectors = index_pages(model, pages, args.size)
        query = embed(model, [args.ask])[0]
        hits = search(chunks, vectors, query)
        with serve(server_binary(), model_file("Q8_0"), 4096) as (url, _):
            result = answer(url, model, args.ask, query, pages, hits)
            print(f"p. {result['page']}: {result['answer']}" if result["answer"] else "No supporting answer selected.")
        for hit in hits:
            print(f"p. {hit['page']}: {hit['text']}")
        return
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    for q in questions:
        if q["passage"] not in pages[q["page"] - 1]:
            raise ValueError(f"Missing gold passage for {q['id']}")
    embeddings = embed(model, [q["question"] for q in questions])
    results = dict(source=SOURCE, sha256=DIGEST, pages=len(pages), overlap=OVERLAP, runs={})
    for size in SIZES:
        chunks, vectors = index_pages(model, pages, size)
        records = []
        for q, vector in zip(questions, embeddings, strict=True):
            hits = search(chunks, vectors, vector)
            rank = next((r for r, h in enumerate(hits, 1)
                         if h["page"] == q["page"] and q["passage"] in h["text"]), None)
            records.append(dict(id=q["id"], rank=rank, retrieved=hits))
        correct = sum(r["rank"] is not None for r in records)
        results["runs"][size] = dict(chunks=len(chunks), correct=correct, queries=records)
        print(f"{size} tokens: {correct}/{len(records)}", flush=True)
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.answers:
        answers = []
        with serve(server_binary(), model_file("Q8_0"), 4096) as (url, _):
            for q, row, query in zip(questions, results["runs"][args.size]["queries"], embeddings, strict=True):
                selected = answer(url, model, q["question"], query, pages, row["retrieved"])
                answers.append(dict(id=q["id"], size=args.size, **selected))
        (HERE / "answers.json").write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
