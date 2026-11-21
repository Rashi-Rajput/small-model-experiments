import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
W, H = A4
M = 42
INK = "#152C36"
TEAL = "#086860"
GRAY = "#536772"
LINE = "#CFDAD8"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class Page:
    def __init__(self, pdf, number, section, title):
        self.pdf = pdf
        self.y = H - 42
        self.width = W - 2 * M
        self.label("SMALL MODEL EXPERIMENTS", M, self.y, TEAL)
        pdf.setFillColor(GRAY)
        pdf.setFont("Helvetica", 9)
        pdf.drawRightString(W - M, self.y, section)
        pdf.setStrokeColor(LINE)
        pdf.line(M, self.y - 12, W - M, self.y - 12)
        self.y -= 48
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 24)
        pdf.drawString(M, self.y, title)
        self.y -= 13
        self.number = number

    def label(self, text, x, y, color=INK):
        self.pdf.setFont("Helvetica-Bold", 9)
        self.pdf.setFillColor(color)
        self.pdf.drawString(x, y, text)

    def paragraph(self, text, size=10.3, gap=7):
        p = Paragraph(text, ParagraphStyle("body", fontName="Helvetica", fontSize=size,
                      leading=size * 1.35, textColor=colors.HexColor(INK)))
        _, height = p.wrap(self.width, H)
        p.drawOn(self.pdf, M, self.y - height)
        self.y -= height + gap

    def heading(self, text):
        self.y -= 15
        self.pdf.setFont("Helvetica-Bold", 13)
        self.pdf.setFillColor(TEAL)
        self.pdf.drawString(M, self.y, text)
        self.y -= 7

    def table(self, widths, header, rows, size=9.2, pad=4):
        widths = [self.width * w for w in widths]
        for i, row in enumerate([header, *rows]):
            cells = []
            for j, (value, w) in enumerate(zip(row, widths, strict=True)):
                style = ParagraphStyle("cell", fontName="Helvetica-Bold" if i == 0 or j == 0 else "Helvetica",
                                       fontSize=size, leading=size * 1.20, alignment=TA_LEFT,
                                       textColor=colors.white if i == 0 else colors.HexColor(INK))
                p = Paragraph(escape(str(value)), style)
                cells.append((p, p.wrap(w - 12, H)[1]))
            height = max(h for _, h in cells) + 2 * pad
            self.pdf.setFillColor(TEAL if i == 0 else "#F0F5F4" if i % 2 else "#FFFFFF")
            self.pdf.rect(M, self.y - height, self.width, height, stroke=0, fill=1)
            x = M + 6
            for (p, h), w in zip(cells, widths, strict=True):
                p.drawOn(self.pdf, x, self.y - pad - h)
                x += w
            self.y -= height
        self.y -= 7

    def end(self, evidence):
        if self.y < 38:
            raise ValueError(f"Page {self.number} overflows by {38 - self.y:.1f} pt")
        self.pdf.setStrokeColor(LINE)
        self.pdf.line(M, 37, W - M, 37)
        self.pdf.setFont("Helvetica", 8)
        self.pdf.setFillColor(GRAY)
        self.pdf.drawString(M, 23, evidence)
        self.pdf.drawRightString(W - M, 23, f"{self.number} / 4")
        self.pdf.showPage()


def classification(pdf):
    r = load("task1_lora/results.json")
    m = load("task1_lora/memory.json")
    p = Page(pdf, 1, "Task 1  /  Low-rank adaptation", "Routing banking requests")
    p.paragraph(f"Can a small encoder tell a delayed transfer from a declined payment? I selected 12 related "
                f"Banking77 intents, sampled {r['train_size']} training requests and held out {r['test_size']} "
                "from the official test split. Each intent had equal representation. I removed exact text "
                "duplicates after normalising whitespace and case.")
    n = r["test_size"]
    p.table([.42, .30, .28], ["Same test set", "Correct / tested", "Top-1 accuracy"],
            [(name, f"{r[key]} / {n}", f"{r[key]/n:.1%}") for name, key in
             [("MiniLM base", "base_correct"), ("MiniLM + LoRA", "tuned_correct")]])
    p.paragraph("Both classifiers picked the nearest intent description by cosine similarity. I kept the "
                "description vectors fixed, so the change in accuracy came from adapting the request encoder.")
    p.heading("What changed during training")
    p.paragraph(f"The base had {r['base_parameters']:,} parameters. Rank 8 adapters on query and value "
                f"added {r['adapter_parameters']:,} trainable weights. These two projections let attention "
                "change which words it follows and what information it carries. A small rank kept CPU training "
                "practical. Alpha 16 scaled the update by 16/8 = 2. I used AdamW at 0.0005 and four epochs "
                "to give the new weights several passes over a small dataset. These were fixed choices.")
    p.heading("Three held-out mistakes")
    p.table([.34, .30, .36], ["Request", "Gold / predicted", "Likely explanation"], [
        ("Why did I get a fee?", "ATM fee / card payment fee", "The request never says where the fee appeared. Either route is plausible."),
        ("My cash withdrawal was partly declined", "Wrong cash amount / declined withdrawal", "The word declined is explicit. Partly carries the clue that some cash was dispensed."),
        ("How do I know when my card will arrive?", "Card arrival / delivery estimate", "The model read when as a request for a forecast. The annotation treats it as tracking an existing order.")], 9.2)
    p.heading("A 7B training budget in decimal GB")
    p.paragraph("Assume BF16 weights and gradients, an FP32 master copy and two FP32 Adam moments. "
                "Each trained parameter costs 2 + 2 + 4 + 4 + 4 = 16 bytes. For square query and value "
                "projections, 32 layers × 2 projections × 8 × (4,096 + 4,096) = 4,194,304 adapter weights. "
                "Budget an illustrative 4 GB for activations and temporary buffers with a small "
                "microbatch and checkpointing. Context length and kernels can change this allowance.", 9.6)
    totals = m["estimated_total_gb"]
    p.table([.18, .65, .17], ["Update", "Model state + assumed temporary memory", "Total"], [
        ("Full", "7B × 16 bytes + 4 GB", f"{totals['full']:.1f} GB"),
        ("LoRA", "7B × 2 bytes + 4,194,304 × 16 bytes + 4 GB", f"{totals['lora']:.1f} GB"),
        ("QLoRA", "7B × (0.5 + 2/64) bytes + adapter state + 4 GB", f"{totals['qlora']:.1f} GB")])
    p.paragraph("The QLoRA estimate assumes packed 4-bit weights and one 2-byte scale per 64 weights. "
                "These are planning calculations. The measured training run used MiniLM on CPU.", 9.2)
    p.end("Evidence in task1_lora/results.json, adapter.safetensors and memory.json")


def retrieval(pdf):
    r = load("task2_retrieval/results.json")
    q = load("task2_retrieval/questions.json")
    p = Page(pdf, 2, "Task 2  /  Retrieval and reading", "Finding evidence in HTTP Semantics")
    p.paragraph(f"I indexed all {r['pages']} PDF pages of RFC 9110, including the contents and index. I removed "
                "running headers and footers, kept page boundaries, and embedded overlapping WordPiece windows "
                "with the original MiniLM. NumPy ranked their unit vectors by dot product.")
    p.heading("How often did a complete passage surface?")
    rows = []
    for size, run in r["runs"].items():
        pages = sum(any(h["page"] == g["page"] for h in row["retrieved"])
                    for row, g in zip(run["queries"], q, strict=True))
        rows.append((size, f"{run['chunks']:,}", f"{pages}/{len(q)}", f"{run['correct']}/{len(q)}",
                     f"{run['correct']/len(q):.1%}"))
    p.table([.18, .16, .22, .24, .20], ["Tokens", "Chunks", "Page in top 5", "Gold span in top 5", "Recall@5"], rows)
    p.paragraph("I counted one hit when a single returned chunk contained the whole labelled passage on its "
                "annotated page. Both sizes shared 16 tokens of overlap. A nearby paragraph can give a useful "
                "answer and still miss this exact-evidence test.")
    p.paragraph("The larger windows won by one question. For Q09, the short windows split the overhead and "
                "privacy explanation across two retrieved chunks. The longer window held both in its first hit. "
                "The trade-off also ran the other way. On Q05, a short chunk isolated the lost-update example "
                "on page 65, while the longer search returned other discussions of entity tags.")
    p.heading("Two failures and what I would change")
    p.paragraph("<b>Q09 · Incomplete evidence.</b> The small-window search reached page 97 at ranks one and "
                "five. A sentence-aware boundary or expansion into the adjacent chunk would join the two "
                "halves before passing evidence to the reader.")
    p.paragraph("<b>Q12 · Right topic, wrong section.</b> Both sizes ranked page 68 first for the question "
                "about checking tags before dates. That page explains validator properties. The reason for "
                "the evaluation order appears on page 115. Indexing section headings with each chunk and "
                "re-ranking for the ordering relationship could bring the precedence section forward.")
    p.heading("What the reader received")
    p.paragraph("I embedded complete text spans inside the five hits and shortlisted up to four. Local "
                "SmolLM2 then chose a span or abstained. A constrained numeric choice let the program return "
                "the original wording and source page. On Q01 it selected the sentence on page 20 that "
                "answers the question. On Q02 it chose the definition of bad whitespace on page 36 and missed "
                "the required parser action. Passage recall and answer quality measure different steps.")
    p.heading("Where lexical search and re-ranking help")
    p.paragraph("BM25 rewards exact query words, especially uncommon ones, and accounts for document length. "
                "That gives an identifier such as If-Range a stronger vote when many passages have similar "
                "embeddings. A re-ranker reads each question together with a candidate and reorders the "
                "shortlist. It can distinguish a definition from a requested action, provided the right "
                "passage was retrieved in the first place.")
    p.end("Evidence in task2_retrieval/questions.json, results.json and answers.json")


def questions(pdf):
    q = load("task2_retrieval/questions.json")
    r = load("task2_retrieval/results.json")
    p = Page(pdf, 3, "Task 2  /  Test set", "The questions and their evidence")
    p.paragraph("Each answer below paraphrases the labelled passage. The page number refers to the RFC PDF. "
                "The final columns give the rank of the first complete gold passage for each window size. "
                "A dash marks a miss in the top five.")
    keys = ["Yes. A user agent can operate without direct human interaction.",
            "Parse and remove the bad whitespace before interpreting the element.",
            "Use the most recent past year with those same final digits.",
            "Treat it as application/octet-stream or inspect the bytes to infer the type.",
            "Losing another editor's update.", "Use HEAD.", "Retry without the 100-continue expectation.",
            "Seconds.", "Repeated disclosure costs bandwidth and can compromise privacy.", "*/*",
            "Vary extends the cache key.", "Entity tags are considered more accurate than dates.",
            "It could overwrite the whole representation.", "307 Temporary Redirect.",
            "A Retry-After field indicates when to try again."]
    rows = []
    for i, (g, key) in enumerate(zip(q, keys, strict=True)):
        ranks = [run["queries"][i]["rank"] or "-" for run in r["runs"].values()]
        rows.append((g["id"], g["question"], key, g["page"], *ranks))
    p.table([.075, .40, .325, .08, .06, .06], ["ID", "Question", "Reference answer", "Page", "64", "192"],
            rows, size=8.9, pad=4.2)
    p.end("Exact source wording and page annotations in task2_retrieval/questions.json")


def inference(pdf):
    r = load("task3_local/results.json")
    c = load("task3_local/capacity.json")
    p = Page(pdf, 4, "Task 3  /  Local inference", "What changed at lower precision?")
    p.paragraph("I ran SmolLM2-135M-Instruct in llama.cpp on an i7-1165G7 CPU. Each setting used four threads, "
                "one slot, a 1,024-token context and greedy decoding capped at 64 new tokens. I warmed a fresh "
                "server for each pass and reversed the order on the second pass.", 9.8)
    rows = []
    for k, s in r["summary"].items():
        run = next(x for x in r["runs"] if x["quant"] == k)
        rows.append((k, f"{run['file_bytes']/2**20:.1f} MiB", f"{s['tokens_per_second']:.1f}",
                     f"{s['peak_mib']:.1f} MiB", f"{s['first_passes']}/10"))
    p.table([.18, .19, .22, .22, .19], ["Weights", "GGUF file", "Decode tok/s", "Peak RSS", "Strict pass"], rows)
    p.paragraph("Speed is total generated tokens divided by summed server decode seconds across both passes. "
                "It excludes prompt processing. Peak is the largest server working set sampled every 25 ms "
                "during the warm workload, in host RAM. Each setting repeated all ten outputs exactly.", 9.5)
    p.heading("What the ten answers actually said")
    p.table([.22, .39, .39], ["Prompt", "Q4_K_M", "Q8_0"], [
        ("Copy parcel code", "Correct code in quotation marks", "Correct code with trailing period"),
        ("Add bead counts", "Repeated both counts without adding", "Same"),
        ("Sort three integers", "Kept the original order", "Same"),
        ("Extract arrival time", "Added Houston and both times", "Repeated the whole schedule"),
        ("Label a review", "Correct negative label in a sentence", "Same"),
        ("Produce JSON", "Wrote Python and hit the token cap", "Also wrote Python and hit the cap"),
        ("Find youngest", "Said Mina was younger than Jo", "Named Mina as youngest"),
        ("Missing time", "Invented 10:00 AM", "Repeated Friday, omitted UNKNOWN"),
        ("Convert units", "Repeated the length in metres", "Same"),
        ("Form initials", "Returned only the first initial", "Repeated the words in title case")], 8.7, 2.7)
    p.paragraph("Both models missed every strict output contract. The code-copy and sentiment responses "
                "retain the right content. The invented opening time is a more serious error than punctuation. "
                "These ten prompts give little basis for a general quality ranking.", 9.6)
    p.heading("Memory and a plan for 20 simultaneous users")
    p.paragraph("Quantisation packs approximate weight values into fewer bits and stores scales for groups. "
                "Other tensors and runtime buffers still occupy memory. The KV cache holds the keys and "
                "values produced at each attention layer for earlier tokens, so decoding can reuse them. "
                "It grows with the combined sequence lengths of active users.", 9.6)
    p.paragraph(f"For the tested Q4 model, the local runtime reported {c['layers']} layers, {c['kv_heads']} KV "
                f"heads, width {c['key_width']} and FP16 cache entries. Keys + values cost 2 × 30 × 3 × 64 × 2 "
                f"= {c['bytes_per_token']:,} bytes/token, matching {c['observed_cache_mib']:.1f} MiB at 1,024 tokens. "
                "Assume 20 active slots, each capped at 1,792 input and 256 output tokens. The cache needs "
                f"20 × 2,048 × 23,040 = {c['cache_bytes']/1e9:.3f} GB. With {c['file_bytes']/1e9:.3f} GB for "
                f"weights and an assumed 4 GB runtime reserve, the budget is {c['estimated_total_gb']:.2f} GB "
                "on a 24 GB GPU.", 9.6)
    p.paragraph("Load one model on the GPU, use continuous batching, stream responses, and queue arrivals "
                "once all slots are occupied. Before promising a latency target, replay simultaneous requests "
                "and measure time to first token and decode speed per user on that GPU.", 9.6)
    p.end("Evidence in task3_local/prompts.json, results.json and capacity.json")


if __name__ == "__main__":
    pdf = Canvas(str(ROOT / "report.pdf"), pagesize=A4)
    pdf.setTitle("Small model experiments")
    for section in (classification, retrieval, questions, inference):
        section(pdf)
    pdf.save()
