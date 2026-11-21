# Small model experiments

[Read the four-page report](report.pdf) for the results, error analysis and memory calculations. All three experiments ran locally on an Intel Core i7-1165G7 using CPU inference and training.

| Experiment | Inputs and implementation | Saved evidence |
| --- | --- | --- |
| Banking intent classification | [LoRA trainer and label descriptions](task1_lora/train.py) | [All held-out predictions](task1_lora/results.json), [trained adapter](task1_lora/adapter.safetensors), [7B calculation](task1_lora/memory.json) |
| Question answering over HTTP Semantics | [Index, search and answer selection](task2_retrieval/retrieve.py), [questions with exact gold passages](task2_retrieval/questions.json) | [Top five chunks for each question and size](task2_retrieval/results.json), [reader selections and alternatives](task2_retrieval/answers.json) |
| Local quantisation comparison | [Benchmark](task3_local/benchmark.py), [prompts and expected outputs](task3_local/prompts.json) | [Raw responses, timing and memory](task3_local/results.json), [serving calculation](task3_local/capacity.json) |

## Reproduce the runs

Use Python 3.14, Git and llama.cpp build b12608. Run the commands from the repository root. On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m task1_lora.train
python -m task1_lora.memory
python -m task3_local.download
curl.exe -fLsS https://github.com/ggml-org/llama.cpp/releases/download/b12608/llama-b12608-bin-win-cpu-x64.zip -o .cache/llama.zip
Expand-Archive .cache/llama.zip -DestinationPath .cache/llama -Force
python -m task2_retrieval.retrieve --answers
python -m task3_local.benchmark
```

To regenerate the report from the saved runs, install `reportlab==5.1.0` and run `python report/build.py`.

On another OS, install the matching [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b12608) and set `LLAMA_SERVER` to the executable path. The GGUF downloader and PDF indexer check file hashes. The first run downloads the encoder, weights and public data. Indexes are cached locally. Running an experiment again replaces its result file.

```shell
python -m task2_retrieval.retrieve --ask "What unit is used for a numeric retry delay?"
```

## Reading the results

The classifier compares each request to the same frozen intent-description vectors before and after training. Its train and test samples come from separate official splits. The adapter contains only the new query and value weights. Reconstruct the classifier with the pinned backbone and label descriptions in the trainer.

The retriever uses the unmodified encoder. A hit requires the entire gold passage on its annotated PDF page in a single top-five chunk. An alternative passage may give a correct answer while missing this test. The reader ranks complete in-chunk text spans by cosine similarity, then asks the local model to select one or abstain. Every returned answer can be checked against its page.

The benchmark starts fresh server processes in Q4, Q8, Q8, Q4 order. Each receives a warm-up before the same ten prompts. Throughput pools server-reported generated tokens over decode time. Process RSS includes weights, KV cache and runtime allocations. The format score accepts surrounding whitespace, insignificant JSON formatting and spaces around list separators. Semantically correct extra prose still fails an answer-only instruction.

## Sources

- [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data) by Casanueva et al. The saved examples are a selected subset of the original evaluation split. [Paper](https://aclanthology.org/2020.nlp4convai-1.5/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41), Apache 2.0. Used as the base encoder for both classification and retrieval.
- [RFC 9110, HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.pdf), R. Fielding, M. Nottingham and J. Reschke. Excerpts are identified by PDF page. Copyright IETF Trust and the document authors. [IETF Trust provisions](https://trustee.ietf.org/license-info/)
- [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), Apache 2.0, using [bartowski's GGUF conversions](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/tree/09816acd5d99df7be770d85ea30822623dab342c).
