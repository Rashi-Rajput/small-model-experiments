import json
import platform
import subprocess
import threading
from pathlib import Path

import psutil

from local_model import MODEL, chat, model_file, post, serve, server_binary


HERE = Path(__file__).resolve().parent
ORDER = ("Q4_K_M", "Q8_0", "Q8_0", "Q4_K_M")
SPEED_PROMPT = "Write a travel diary about a walk along the coast."


def passes(case, text):
    text = text.strip()
    if case["id"] == "json":
        try:
            parsed = json.loads(text)
        except ValueError:
            return False
        return parsed == case["expected"]
    if case["id"] == "sort":
        text = ",".join(x.strip() for x in text.split(","))
    return text == case["expected"]


def measure(quant, prompts):
    model = model_file(quant)
    with serve(server_binary(), model, 1024) as (url, server):
        process = psutil.Process(server.pid)
        chat(url, [dict(role="user", content="Say hello.")], 12)
        samples = [process.memory_info().rss]
        done = threading.Event()

        def sample_memory():
            while not done.wait(0.025):
                samples.append(process.memory_info().rss)

        sampler = threading.Thread(target=sample_memory)
        sampler.start()
        rows = []
        try:
            for case in prompts:
                response = chat(url, [dict(role="user", content=case["prompt"])], 64)
                choice = response["choices"][0]
                text = choice["message"]["content"]
                timings = response["timings"]
                rows.append(dict(id=case["id"], text=text, passed=passes(case, text),
                                 finish=choice["finish_reason"], tokens=timings["predicted_n"],
                                 decode_ms=timings["predicted_ms"], prompt_ms=timings["prompt_ms"]))
        finally:
            done.set()
            sampler.join()
        control = post(url + "/completion", dict(prompt=SPEED_PROMPT, n_predict=128, ignore_eos=True,
                       temperature=0, seed=41, cache_prompt=False))["timings"]
        if control["predicted_n"] != 128:
            raise ValueError("Fixed-length decode ended early")
        return dict(quant=quant, file_bytes=model.stat().st_size, warm_rss=samples[0],
                    peak_rss=max(samples), samples=len(samples), control=control, outputs=rows)


def main():
    prompts = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
    version = subprocess.run([server_binary(), "--version"], capture_output=True, text=True, check=True)
    results = dict(model=MODEL, platform=platform.platform(), cpu=platform.processor(), physical_cores=psutil.cpu_count(logical=False),
                   runtime=(version.stdout + version.stderr).strip(), threads=4, context=1024, max_tokens=64,
                   temperature=0, seed=41, rss_interval_ms=25, control_prompt=SPEED_PROMPT, runs=[], summary={})
    for quant in ORDER:
        run = measure(quant, prompts)
        results["runs"].append(run)
        print(quant, sum(r["passed"] for r in run["outputs"]), "/", len(prompts), flush=True)
    for quant in dict.fromkeys(ORDER):
        runs = [r for r in results["runs"] if r["quant"] == quant]
        rows = [row for r in runs for row in r["outputs"]]
        tokens = sum(r["tokens"] for r in rows)
        seconds = sum(r["decode_ms"] for r in rows) / 1000
        control_speed = sum(r["control"]["predicted_n"] for r in runs) * 1000 / sum(
            r["control"]["predicted_ms"] for r in runs)
        results["summary"][quant] = dict(tokens=tokens, decode_seconds=seconds, tokens_per_second=tokens / seconds,
                                        control_tokens_per_second=control_speed,
                                        peak_mib=max(r["peak_rss"] for r in runs) / 2**20,
                                        first_passes=sum(r["passed"] for r in runs[0]["outputs"]),
                                        repeated_outputs=sum(a["text"] == b["text"] for a, b in
                                                             zip(runs[0]["outputs"], runs[1]["outputs"], strict=True)))
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(results["summary"])


if __name__ == "__main__":
    main()
