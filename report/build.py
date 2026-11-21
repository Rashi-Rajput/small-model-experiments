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


def generation(pdf):
    r = load("task1_lora/results.json")
    m = load("task1_lora/memory.json")
    settings = r["settings"]
    p = Page(pdf, 1, "Task 1  /  Generative LoRA", "From stock notes to JSON")
    p.paragraph(f"SmolLM2-135M-Instruct trained on {r['train_size']} synthetic inventory notes. Each target "
                f"has an action, a SKU and an integer quantity. All {r['test_size']} test notes use held-out "
                "templates and target records. Half contain unseen SKUs.")
    n = r["test_size"]
    p.table([.37, .24, .23, .16], ["Checkpoint", "Schema valid", "All fields right", "Exact rate"],
            [(label, f"{r[name + '_schema']}/{n}", f"{r[name + '_correct']}/{n}", f"{r[name + '_correct']/n:.1%}")
             for name, label in [("base", "Original checkpoint"), ("tuned", "With trained LoRA")]])
    capped = sum(not row["base"]["ended"] for row in r["predictions"])
    p.paragraph(f"Both checkpoints generated greedily from identical prompts. {capped}/{n} base replies hit "
                f"the {settings['max_new_tokens']}-token cap. Exact credit requires the whole reply to parse "
                "with the right keys, types and values, allowing different whitespace or key order.", 9.5)
    example = next((row for row in r["predictions"] if row["tuned"]["correct"]), r["predictions"][0])
    p.paragraph("<b>Actual generation</b> &nbsp; " + escape(example["request"]) +
                "<br/><b>Base</b> &nbsp; " + escape(example["base"]["text"]) +
                "<br/><b>LoRA</b> &nbsp; " + escape(example["tuned"]["text"]), 9.4)
    issue = [row for row in r["predictions"] if row["target"]["action"] == "issue"]
    copied = sum(row["tuned"]["schema"] and json.loads(row["tuned"]["text"])["sku"] == row["target"]["sku"]
                 for row in r["predictions"])
    check = load("task1_lora/quantity_check.json")
    p.paragraph(f"The adapter copied {copied}/{n} SKUs, but got {sum(row['tuned']['correct'] for row in issue)}"
                f"/{len(issue)} outgoing records right. In the main data, confirmed quantities are always "
                "smaller than competing figures. On separate paired notes that swap those numbers, it got "
                f"both versions right in {check['both_correct']}/{check['pairs']} pairs.", 9.4)
    p.heading("What the adapter learned from")
    p.paragraph(f"The base decoder had {r['base_parameters']:,} parameters. Rank {settings['rank']} "
                f"query and value adapters added {r['adapter_parameters']:,} trainable weights. Their small "
                "size kept CPU training feasible while allowing attention to follow item codes and quantities. "
                f"Alpha {settings['alpha']} scales the update by {settings['alpha']/settings['rank']:g}. "
                f"AdamW at {settings['lr']} and {settings['epochs']} passes gave the new weights repeated, small "
                "updates. Only next-token errors on the JSON and end-of-turn token contributed to the loss.", 9.7)
    p.heading("Three held-out generation errors")
    records = {row["request"]: row for row in r["predictions"]}
    mistakes = []
    for failure in load("task1_lora/failures.json"):
        row = records[failure["request"]]
        if row["tuned"]["correct"] or row["tuned"]["text"] != failure["generated"]:
            raise ValueError("Failure analysis does not match the saved generation")
        mistakes.append((row["request"], failure["difference"], failure["explanation"]))
    p.table([.39, .24, .37], ["Request", "Expected / generated", "Likely explanation"], mistakes, 8.7, 3)
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
                "These are planning calculations. The measured run trained SmolLM2 on CPU.", 9.2)
    p.paragraph("AI drafted the test questions and report prose. The measurements come from runs on the "
                "local machine.", 9.2)
    p.end("Evidence in task1_lora/results.json, adapter.safetensors and memory.json")


def retrieval(pdf):
    r = load("task2_retrieval/results.json")
    q = load("task2_retrieval/questions.json")
    p = Page(pdf, 2, "Task 2  /  Retrieval and reading", "Finding evidence in HTTP Semantics")
    p.paragraph(f"The corpus covers all {r['pages']} pages of RFC 9110, including its contents and index. "
                "MiniLM embeds overlapping windows of text from each page. NumPy ranks them by cosine "
                "similarity. For the five highest-scoring windows, the revised system completes the text at "
                "each boundary and includes one neighbouring span on each side when available.")
    p.heading("How often did a complete passage surface?")
    rows = []
    for size, run in r["runs"].items():
        pages = sum(any(h["page"] == g["page"] for h in row["retrieved"])
                    for row, g in zip(run["queries"], q, strict=True))
        seeds = sum(row["seed_rank"] is not None for row in run["queries"])
        rows.append((size, f"{pages}/{len(q)}", f"{seeds}/{len(q)}", f"{run['correct']}/{len(q)}",
                     f"{run['correct']/len(q):.1%}"))
    p.table([.17, .19, .21, .22, .21], ["Seed tokens", "Page in top 5", "Gold in seed", "After expansion", "Final recall@5"], rows)
    p.paragraph("A hit requires the full labelled passage on the annotated page in one of the first five "
                "results. Both seed sizes use 16 tokens of overlap. The 192-token seeds had a one-question "
                "edge before expansion. Afterwards the scores tied. For Q09 the smaller seed had cut the "
                "privacy explanation in two. Expanding the text recovered it without changing its page rank.")
    p.heading("Two failures after expansion")
    p.paragraph("<b>Q05 · Specific use of an entity tag.</b> The 192-token search ranked a discussion on "
                "page 69 first and omitted the lost-update example on page 65. The 64-token search did find "
                "that example. Merging candidates from both sizes before ranking could recover it.")
    p.paragraph("<b>Q12 · Right topic, wrong section.</b> Both sizes ranked page 68 first for the question "
                "about checking tags before dates. That page explains validator properties. The reason for "
                "the evaluation order appears on page 115. Indexing section headings with each chunk and "
                "re-ranking for the ordering relationship could bring the precedence section forward.")
    p.heading("Which evidence reached the final answer?")
    answers = load("task2_retrieval/answers.json")
    offered = sum(a["gold_offered"] for a in answers)
    selected = sum(a["gold_selected"] for a in answers)
    previous = load("task2_retrieval/reader_comparison.json")
    readers = []
    for key, label in [("original", "SmolLM2 · four numbered choices"),
                       ("numbered_choices", "Qwen · choose by number"),
                       ("answer_and_quote", "Qwen · generate answer and quote")]:
        readers.append((label, f"{sum(a['offered'] for a in previous[key])}/{len(q)}",
                        f"{sum(a['matched'] for a in previous[key])}/{len(q)}"))
    readers.append(("Qwen draft + evidence ranking", f"{offered}/{len(q)}", f"{selected}/{len(q)}"))
    p.table([.44, .28, .28], ["Reader on 192-token seeds", "Gold offered", "Gold selected"], readers)
    p.paragraph("The final reader uses a Qwen2.5-1.5B-Instruct draft as a search hint. MiniLM compares each "
                "span with the draft and question. Shared, uncommon draft terms add to its score. The reader "
                "returns the highest-scoring original passage and page, or abstains on UNKNOWN. All Qwen "
                "runs used the same spans. The questions had already been reviewed during development.")
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
    review = load("task2_retrieval/answer_review.json")
    answers = load("task2_retrieval/answers.json")
    p = Page(pdf, 3, "Task 2  /  Test set", "The questions and their evidence")
    p.paragraph("Each reference answer paraphrases the labelled passage on the indicated PDF page. "
                "The 64 and 192 columns rank the first expanded hit containing that passage. A dash marks "
                "a miss. QA grades whether the final 192-token reader's returned quotation answers the question.")
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
    for i, (g, key, answer, assessment) in enumerate(zip(q, keys, answers, review["items"], strict=True)):
        if (g["id"], answer["draft"], answer["answer"]) != (
                assessment["id"], assessment["draft"], assessment["evidence"]):
            raise ValueError("The answer review refers to a different reader response")
        ranks = [run["queries"][i]["rank"] or "-" for run in r["runs"].values()]
        rows.append((g["id"], g["question"], key, g["page"], *ranks, assessment["grade"]))
    p.table([.071, .391, .295, .067, .054, .059, .063], ["ID", "Question", "Reference answer", "Page", "64", "192", "QA"],
            rows, size=8.9, pad=4.2)
    grades = [row["grade"] for row in review["items"]]
    p.paragraph(f"Y = complete and supported ({grades.count('Y')}). P = partial or incompletely supported "
                f"({grades.count('P')}). N = wrong or unsupported ({grades.count('N')}). "
                f"A = abstained ({grades.count('A')}). Equivalent evidence from another page earns full credit "
                "when it supports the whole answer.", 9.3)
    p.paragraph(review["observation"], 9.3)
    p.end("Source annotations in questions.json and per-answer judgements in answer_review.json")


def inference(pdf):
    r = load("task3_local/results.json")
    c = load("task3_local/capacity.json")
    review = load("task3_local/quality.json")
    p = Page(pdf, 4, "Task 3  /  Local inference", "What changed at lower precision?")
    p.paragraph(f"{r['model']} ran in llama.cpp on an i7-1165G7 CPU. Each setting used four threads, "
                "one slot, a 1,024-token context and greedy decoding capped at 64 new tokens. The benchmark "
                "warmed a fresh server for each pass and reversed the order on the second pass.", 9.8)
    rows = []
    for k, s in r["summary"].items():
        run = next(x for x in r["runs"] if x["quant"] == k)
        for item, pred in zip(review["items"], run["outputs"], strict=True):
            if item["id"] != pred["id"] or item[k]["text"] != pred["text"]:
                raise ValueError("The quality review refers to a different benchmark response")
        content = sum(item[k]["content_correct"] for item in review["items"])
        rows.append((k, f"{s['tokens_per_second']:.1f}", f"{s['control_tokens_per_second']:.1f}",
                     f"{s['peak_mib']:.1f}", f"{s['first_passes']}/10", f"{content}/10"))
    p.table([.16, .18, .19, .19, .14, .14], ["Weights", "Task tok/s", "Fixed tok/s", "Peak MiB", "Strict", "Content"], rows)
    p.paragraph("Speed pools generated tokens over server decode time across two passes. Fixed tok/s uses "
                "the same prompt and exactly 128 new tokens per pass. Peak RSS samples host RAM every 25 ms "
                "during the ten questions. Strict scoring checks the requested format. Content scoring credits "
                "an unambiguous correct answer despite extra formatting or prose.", 9.5)
    p.heading("What the ten answers actually said")
    p.table([.23, .385, .385], ["Prompt", "Q4_K_M", "Q8_0"],
            [(item["label"], item["Q4_K_M"]["observation"], item["Q8_0"]["observation"])
             for item in review["items"]], 8.7, 2.7)
    p.paragraph(review["interpretation"], 9.6)
    p.heading("Memory and a plan for 20 simultaneous users")
    files = {run["quant"]: run["file_bytes"] / 1e9 for run in r["runs"]}
    p.paragraph("Quantisation packs approximate weight values into fewer bits and stores scales for groups. "
                f"The Q4 file used {files['Q4_K_M']:.2f} GB on disk and Q8 used {files['Q8_0']:.2f} GB. "
                "RSS includes resident model pages and runtime buffers. The KV cache holds the keys and "
                "values produced at each attention layer for earlier tokens, so decoding can reuse them. "
                "It grows with the combined sequence lengths of active users.", 9.6)
    p.paragraph(f"For the tested Q4 model, the local runtime reported {c['layers']} layers, {c['kv_heads']} KV "
                f"heads, width {c['key_width']} and FP16 cache entries. Keys + values cost "
                f"2 × {c['layers']} × {c['kv_heads']} × {c['key_width']} × 2 "
                f"= {c['bytes_per_token']:,} bytes/token, matching {c['observed_cache_mib']:.1f} MiB at 1,024 tokens. "
                "Assume 20 active slots, each capped at 1,792 input and 256 output tokens. The cache needs "
                f"20 × 2,048 × {c['bytes_per_token']:,} = {c['cache_bytes']/1e9:.3f} GB. With {c['file_bytes']/1e9:.3f} GB for "
                f"weights and an assumed 4 GB runtime reserve, the budget is {c['estimated_total_gb']:.2f} GB "
                "on a 24 GB GPU.", 9.6)
    p.paragraph("Load one model on the GPU, use continuous batching, stream responses, and queue arrivals "
                "once all slots are occupied. Before promising a latency target, replay simultaneous requests "
                "and measure time to first token and decode speed per user on that GPU.", 9.6)
    p.end("Evidence in task3_local/prompts.json, results.json and capacity.json")


if __name__ == "__main__":
    pdf = Canvas(str(ROOT / "report.pdf"), pagesize=A4)
    pdf.setTitle("Small model experiments")
    for section in (generation, retrieval, questions, inference):
        section(pdf)
    pdf.save()
