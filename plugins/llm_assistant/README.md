# Optional LLM assistant plugin

NeuroSTIAS works fully without this plugin. When it is installed, `neurostias ask` answers
questions by calling NeuroSTIAS skills through a model of your choice.

```bash
pip install -e '.[llm]'
neurostias ask "Which cortical layers are enriched for SST interneurons?" --data my.h5ad
neurostias ask "..." --model qwen/qwen3.6-27b                                    # another local model
neurostias ask "..." --provider local --endpoint http://localhost:11434/v1 --model llama3.1   # Ollama
NEUROSTIAS_ALLOW_CLOUD=1 neurostias ask "..." --provider anthropic               # Claude (ANTHROPIC_API_KEY)
```

| Provider | Backend | Where data goes |
|---|---|---|
| `local` (default) | any OpenAI-compatible server on localhost (LM Studio, Ollama, llama.cpp, vLLM) | stays on this machine; non-localhost endpoints are refused |
| `anthropic` | official `anthropic` SDK, default `claude-opus-5-5`, effort `medium`, server-side refusal fallback (`fallbacks="default"`) | skill result summaries only (see below) |
| `openai` | official `openai` SDK | skill result summaries only |

**What is sent to a cloud provider:** your question and the JSON summaries returned by skills
(counts, statistics, top-k lists). It never sends expression matrices, per-cell tables or files.
Lists are truncated to 20 items, and your home directory is replaced by `~`. You must acknowledge
cloud use once (`cloud.acknowledged: true` in `configs/llm.yaml`, or `NEUROSTIAS_ALLOW_CLOUD=1`).

**Guardrail:** every number in an answer is checked against the tool results and the question.
Numbers that cannot be traced are listed under the answer as a WARNING. The check covers numbers
only, so wording, gene-name expansions or interpretations can still be wrong. In testing, a 9B
local model expanded "VLMC" incorrectly. Treat prose as a draft and the result files as the
source of truth.

**Local documents:** `ask(..., docs="folder")` adds a `search_docs` tool (BM25 over .txt/.md/.pdf).
Keep copyrighted books in `plugins/llm_assistant/corpus_private/`, which is git-ignored.
