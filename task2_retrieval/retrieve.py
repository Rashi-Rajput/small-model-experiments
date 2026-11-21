import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pymupdf
import torch

from encoder import Encoder
from local_model import CACHE, MODEL, chat, model_file, serve, server_binary


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


def page_spans(page):
    boundaries = [0]
    for match in re.finditer(r'(?<=[.!?;])\s+(?=[A-Z0-9"(])|\s+(?=•\s)', page):
        boundaries.append(match.end())
    spans = []
    for start, stop in zip(boundaries, boundaries[1:] + [len(page)], strict=True):
        words = list(re.finditer(r"\S+", page[start:stop]))
        for i in range(0, len(words), 80):
            last = min(i + 80, len(words)) - 1
            spans.append((start + words[i].start(), start + words[last].end()))
    return spans


def search(chunks, vectors, query, pages, spans):
    scores = vectors @ query
    hits = []
    for i in np.argsort(-scores, kind="stable")[:5]:
        seed = chunks[i]
        number = seed["page"]
        units = spans[number - 1]
        covered = [j for j, (start, end) in enumerate(units) if end > seed["start"] and start < seed["end"]]
        first = max(0, covered[0] - 1)
        last = min(len(units) - 1, covered[-1] + 1)
        start, end = units[first][0], units[last][1]
        hits.append(dict(page=number, score=float(scores[i]), start=start, end=end,
                         text=pages[number - 1][start:end], seed_start=seed["start"], seed_end=seed["end"]))
    return hits


def supporting_span(model, question, draft, choices):
    texts = [text for _, text in choices]
    q, a, *vectors = embed(model, [question, draft, *texts])
    semantic = np.asarray(vectors) @ (0.25 * q + 0.75 * a)
    token_ids = model.tokenizer([draft, *texts], add_special_tokens=False, verbose=False)["input_ids"]
    terms = set(token_ids[0])
    documents = [set(tokens) for tokens in token_ids[1:]]
    frequency = Counter(token for doc in documents for token in doc)
    weights = {t: 1 + np.log((1 + len(documents)) / (1 + frequency[t])) for t in terms}
    denominator = sum(weights.values()) or 1
    overlap = [sum(weights[t] for t in sorted(terms & doc)) / denominator for doc in documents]
    return int(np.argmax(semantic + overlap))


def answer(url, model, question, pages, spans, hits):
    choices = []
    seen = set()
    for hit in hits:
        number = hit["page"]
        for start, end in spans[number - 1]:
            key = (number, start)
            if hit["start"] <= start and end <= hit["end"] and key not in seen:
                seen.add(key)
                choices.append((number, pages[number - 1][start:end]))
    context = "\n".join(text for _, text in choices)
    messages = [dict(role="system", content="Use the supplied HTTP excerpts to answer the question in one "
                     "short sentence. Use the terminology of the source. If they do not contain an answer, "
                     "reply UNKNOWN. Give only the answer."),
                dict(role="user", content=f"{context}\n\nQuestion: {question}")]
    if not choices:
        return dict(draft=None, answer=None, page=None, choices=[])
    draft = chat(url, messages, 80)["choices"][0]["message"]["content"].strip()
    if not draft or re.match(r"^unknown\b", draft, flags=re.IGNORECASE):
        return dict(draft=draft, answer=None, page=None, choices=choices)
    index = supporting_span(model, question, draft, choices)
    page, text = choices[index]
    return dict(draft=draft, answer=text, page=page, choices=choices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask")
    parser.add_argument("--size", type=int, choices=SIZES, default=192)
    parser.add_argument("--answers", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    pages = read_pages()
    spans = [page_spans(page) for page in pages]
    model = Encoder().eval()
    if args.ask:
        chunks, vectors = index_pages(model, pages, args.size)
        query = embed(model, [args.ask])[0]
        hits = search(chunks, vectors, query, pages, spans)
        with serve(server_binary(), model_file("Q4_K_M"), 8192) as (url, _):
            result = answer(url, model, args.ask, pages, spans, hits)
            if result["answer"]:
                print(f"p. {result['page']}: {result['answer']}")
            else:
                print("No supporting answer selected.")
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
            hits = search(chunks, vectors, vector, pages, spans)
            rank = next((r for r, h in enumerate(hits, 1)
                         if h["page"] == q["page"] and q["passage"] in h["text"]), None)
            seed_rank = next((r for r, h in enumerate(hits, 1) if h["page"] == q["page"] and
                              q["passage"] in pages[h["page"] - 1][h["seed_start"]:h["seed_end"]]), None)
            records.append(dict(id=q["id"], seed_rank=seed_rank, rank=rank, retrieved=hits))
        correct = sum(r["rank"] is not None for r in records)
        results["runs"][size] = dict(chunks=len(chunks), correct=correct, queries=records)
        print(f"{size} tokens: {correct}/{len(records)}", flush=True)
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.answers:
        answers = []
        with serve(server_binary(), model_file("Q4_K_M"), 8192) as (url, _):
            for q, row in zip(questions, results["runs"][args.size]["queries"], strict=True):
                selected = answer(url, model, q["question"], pages, spans, row["retrieved"])
                available = any(pg == q["page"] and q["passage"] in s for pg, s in selected["choices"])
                matched = selected["page"] == q["page"] and q["passage"] in (selected["answer"] or "")
                answers.append(dict(id=q["id"], model=MODEL, quant="Q4_K_M", size=args.size,
                                    gold_offered=available, gold_selected=matched, **selected))
                print(q["id"], selected["page"], flush=True)
        (HERE / "answers.json").write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
