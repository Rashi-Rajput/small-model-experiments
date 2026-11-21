# Small model experiments

[Read the four-page report](report.pdf) for the results, error analysis and memory calculations. All three experiments ran locally on an Intel Core i7-1165G7 using CPU inference and training.

Task 1 trains a causal language model to generate inventory records. The [previous submission](https://github.com/Rashi-Rajput/small-model-experiments/tree/49b850603693c8ebc42b6eda1c8e31d223bde1bc) retains the earlier banking classifier. Tasks 2 and 3 use the same retrieval questions and benchmark prompts as before. The questions, report prose and qualitative assessments were AI-drafted.

| Experiment | Inputs and implementation | Saved evidence |
| --- | --- | --- |
| Inventory notes to JSON | [Causal LM training and generation](task1_lora/train.py), [train and test data](task1_lora/dataset.json) | [Raw generated text and scores](task1_lora/results.json), [trained adapter](task1_lora/adapter.safetensors), [three failures](task1_lora/failures.json), [7B calculation](task1_lora/memory.json) |
| Question answering over HTTP Semantics | [Index and reader](task2_retrieval/retrieve.py), [questions with gold passages](task2_retrieval/questions.json) | [Top-five hits](task2_retrieval/results.json), [reader outputs](task2_retrieval/answers.json), [answer and citation review](task2_retrieval/answer_review.json) |
| Local quantisation comparison | [Benchmark](task3_local/benchmark.py), [prompts and expected outputs](task3_local/prompts.json) | [Raw responses, timing and memory](task3_local/results.json), [content and format observations](task3_local/quality.json), [serving calculation](task3_local/capacity.json) |

## Reproduce the runs

Use Python 3.14, Git and llama.cpp build b12608. Run the commands from the repository root. On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m task1_lora.train
python -m task1_lora.train --verify
python -m task1_lora.check_quantities
python -m task1_lora.memory
python -m task3_local.download
curl.exe -fLsS https://github.com/ggml-org/llama.cpp/releases/download/b12608/llama-b12608-bin-win-cpu-x64.zip -o .cache/llama.zip
Expand-Archive .cache/llama.zip -DestinationPath .cache/llama -Force
python -m task2_retrieval.retrieve --answers
python -m task3_local.benchmark
python -m task3_local.capacity
```

To regenerate the report from the saved runs, install `reportlab==5.1.0` and run `python report/build.py`.

On another OS, install the matching [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b12608) and set `LLAMA_SERVER` to the executable path. The GGUF downloader and PDF indexer check file hashes. The first run downloads the encoder, weights and public data. Indexes are cached locally. Running an experiment again replaces its result file.

```shell
python -m task1_lora.train --ask "We received 12 units of BK-14."
python -m task2_retrieval.retrieve --ask "What unit is used for a numeric retry delay?"
```

## Reading the results

The first task uses the published SmolLM2-135M-Instruct checkpoint. The model reads an inventory note and writes a JSON object containing the movement type, SKU and integer quantity. LoRA updates the attention query and value projections. Training predicts each next JSON token, including the end-of-turn token. Prompt tokens and padding contribute no targets to the loss. Both checkpoints generate greedily from the same chat prompt with the same token budget. The scorer parses the entire unedited continuation and checks keys, value types and field values. Key order and whitespace are immaterial. Markdown fences, duplicate keys, extra prose and numeric strings fail the schema check.

The synthetic dataset is included in the repository. Run `python -m task1_lora.data` to regenerate it. The training split has 24 templates and 192 notes. The test split has 12 different templates and 72 notes, with 24 examples per action. Each target record is unique across the two splits. Half the test requests use SKUs absent from training. The test covers physical counts, stock arrivals and issues, with a mixture of single quantities and competing figures. Scores describe these templates. Run `python -m unittest -v task1_lora.test_train` to check split isolation, strict JSON grading, response masking and adapter gradients.

In the main dataset, a confirmed quantity is always smaller than a competing figure. A [separate diagnostic](task1_lora/quantity_check.json) swaps the two numbers in six pairs of new requests. A pair passes only if the adapter generates the complete correct record in both directions. The 12 diagnostic notes are excluded from the main scores.

The retriever embeds 64- and 192-token windows with the original MiniLM encoder. For each of the five highest-scoring windows, it completes the boundary spans and adds one neighbour on either side within the page. Long spans are divided at 80 words. Each record stores the original window offsets and the expanded text. A strict hit requires the whole annotated passage on its source page.

Qwen2.5-1.5B-Instruct Q4 reads all distinct spans and drafts a short answer. UNKNOWN produces an abstention. Otherwise the program ranks the source spans using cosine similarity to the draft (75%) and question (25%), plus the fraction of draft tokens covered, weighted by inverse frequency across the candidates. It returns the winning verbatim span and PDF page. The draft stays in the raw results for auditing. The batch reader uses 192-token seeds by default. An equivalent answer elsewhere in the RFC can miss the strict passage metric. [Earlier reader trials](task2_retrieval/reader_comparison.json) include the numbered-choice and constrained-quotation approaches.

The benchmark starts fresh server processes in Q4, Q8, Q8, Q4 order. Each receives a warm-up before the same ten prompts. Throughput pools server-reported generated tokens over decode time. A separate control generates exactly 128 tokens per pass with the same prompt and end-of-sequence stopping disabled. RSS is sampled only during the ten-prompt workload and includes weights, KV cache and runtime allocations. The format score accepts surrounding whitespace, insignificant JSON formatting and spaces around list separators. The content review credits an unambiguous correct answer even when it includes extra formatting or explanation. Incorrect or conflicting claims fail the content review.

Four retrieval regression checks run with `python -m unittest -v task2_retrieval.test_retrieve`. A separate live check asked which method removes the association between a URI and its current functionality. The system returned the passage for 421 Misdirected Request on page 141. The correct method was DELETE. This extra check is outside the 15-question scores.

## Sources

- [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac), Apache 2.0. Used as the generative Task 1 backbone.
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41), Apache 2.0. Used as the Task 2 retrieval encoder.
- [RFC 9110, HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.pdf), R. Fielding, M. Nottingham and J. Reschke. Excerpts are identified by PDF page. Copyright IETF Trust and the document authors. [IETF Trust provisions](https://trustee.ietf.org/license-info/)
- [Qwen2.5-1.5B-Instruct, official GGUF weights](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9), Apache 2.0. The first experiment used [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
