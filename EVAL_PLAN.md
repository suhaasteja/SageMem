# LoCoMo Evaluation Pipeline — Build Plan

## Goal

Build a reproducible evaluation pipeline that benchmarks **my custom memory system** on the **LoCoMo** dataset using the **Mem0/Memobase methodology** (the de facto standard the field compares against).

The output should be: results JSON per run, a category-broken-down score table, token + latency stats, and a markdown report ready to share or include in a paper.

---

## Background (read this before coding)

**LoCoMo** = "Long Conversation Memory" benchmark from Maharana et al. (ACL 2024). Multi-session synthetic conversations between two speakers, with QA pairs that test whether a system remembers facts across sessions.

The released `locomo10.json` contains:
- **10 conversations**, each with up to ~32 sessions, averaging ~16K tokens total
- **1,986 QA pairs** total across the 10 conversations
- Five QA categories (see below)

**The standard convention** (Mem0 paper, Zep, Memobase, MemMachine, etc.) is to evaluate on **categories 1–4 only**, dropping category 5 (adversarial/unanswerable). This yields **~1,540 QA pairs**. Category 5 has documented ground-truth issues and tests refusal rather than recall.

### QA category mapping (from the LoCoMo paper)

| Category | Type | What it tests |
|---|---|---|
| 1 | Single-hop | Answer comes from a single session |
| 2 | Multi-hop | Synthesize info across multiple sessions |
| 3 | Temporal | Time-based reasoning, ordering, dates |
| 4 | Open-domain | Combine speaker info with world knowledge |
| 5 | Adversarial | Designed to be unanswerable; **EXCLUDE from main eval** |

> Note: some sources list categories in a different order. **Trust the integer `category` field in the dataset, not the position.** Verify by inspecting a few examples per category before running the full eval.

### Reference implementations to mirror

- **Primary reference**: `https://github.com/mem0ai/mem0/tree/main/evaluation` — this is the canonical methodology. Read `evals.py`, `prompts.py`, `generate_scores.py`, and the per-technique implementations in `src/`.
- **Cleaner fork**: `https://github.com/memodb-io/memobase/tree/main/docs/experiments/locomo-benchmark` — easier Makefile workflow, forked from Mem0's harness.
- **Audit / known issues**: `https://github.com/dial481/locomo-audit` — read this to understand judge prompt drift, ground-truth errors, and statistical issues with category sample sizes.

---

## Dataset schema

`locomo10.json` is a JSON array. Each element = one conversation with this shape:

```jsonc
{
  "sample_id": "conv-...",
  "conversation": {
    "speaker_a": "Name",
    "speaker_b": "Name",
    "session_1_date_time": "...",
    "session_1": [
      {"speaker": "...", "dia_id": "D1:1", "text": "...",
       "img_url": "...", "blip_caption": "...", "query": "..." }
      // images are optional; not all turns have them
    ],
    "session_2_date_time": "...",
    "session_2": [ ... ],
    // up to ~session_32
    "session_<N>_observation": { ... },   // generated, optional
    "session_<N>_summary": "..."          // generated, optional
  },
  "event_summary": { ... },               // ground truth for event task — IGNORE for QA
  "qa": [
    {
      "question": "...",
      "answer": "...",
      "category": 1,
      "evidence": ["D2:5", "D4:3"]   // dia_ids that support the answer
      // adversarial (cat 5) items may have "adversarial_answer" instead
    },
    ...
  ]
}
```

**Important parsing notes:**
- Sessions are stored as separate keys (`session_1`, `session_2`, ...) **not** in a list. Iterate by sorting numeric suffixes.
- We are doing **QA only**. Ignore event summarization and multi-modal turn generation.
- Images: do **not** rely on `img_url` (often dead). Use `blip_caption` as text fallback if needed, or skip image-dependent questions entirely (the audit repo has a list of these).

### Where to get the dataset

- Original: `https://github.com/snap-research/locomo` → `data/locomo10.json`
- Mirror with the exact file Mem0 uses: linked from the Mem0 evaluation README's Google Drive link

Place at `dataset/locomo10.json`.

---

## Project structure to create

```
locomo_eval/
├── README.md
├── pyproject.toml              # use uv; deps below
├── .env.example                # OPENAI_API_KEY, MY_MEMORY_API_KEY, etc.
├── dataset/
│   └── locomo10.json           # not committed; document download
├── src/
│   ├── __init__.py
│   ├── config.py               # models, top_k, paths, seeds
│   ├── data.py                 # dataset loader, schema validation
│   ├── prompts.py              # answer prompt + judge prompt (canonical)
│   ├── judge.py                # LLM-as-judge wrapper
│   ├── metrics.py              # F1, BLEU-1, token counting
│   ├── adapters/
│   │   ├── base.py             # MemoryAdapter ABC
│   │   ├── my_system.py        # adapter for MY memory system (the one under test)
│   │   ├── full_context.py     # baseline: stuff entire convo into prompt
│   │   └── naive_rag.py        # baseline: chunk + embed + top-k
│   ├── pipeline/
│   │   ├── ingest.py           # session-by-session ingestion
│   │   ├── query.py            # ask each QA, capture answer + tokens + latency
│   │   └── evaluate.py         # run judge over results
│   └── reporting/
│       ├── aggregate.py        # mean ± std across runs, per category
│       └── report.py           # writes report.md + results.csv
├── scripts/
│   ├── run_full_eval.py        # entrypoint: --adapter my_system --runs 3
│   └── verify_dataset.py       # SHA + schema + counts sanity check
└── tests/
    ├── test_data.py            # schema, category counts, evidence parsing
    ├── test_metrics.py         # known-answer fixtures
    └── test_judge.py           # judge gives 1 for paraphrases, 0 for wrong
```

---

## Pipeline shape (this is the contract)

For each conversation, for each run:

1. **Ingest** — Reset memory state. Then iterate sessions in chronological order (`session_1`, `session_2`, …). For each session, feed turns to the memory system **in order**, with the session's `date_time` as a metadata hint. Do not bulk-load the whole conversation. LoCoMo is meant to test incremental memory; bulk-loading invalidates the comparison.

2. **Query** — For each QA item in the conversation (categories 1–4 only):
   - Pass the question to the memory system → get retrieved context
   - Send `(retrieved_context, question)` to the answering LLM (default `gpt-4o-mini`) using the canonical answer prompt
   - Capture: predicted answer, retrieved tokens, prompt tokens, completion tokens, search latency, generation latency, total latency

3. **Judge** — For each QA, send `(question, gold_answer, predicted_answer)` to the judge LLM. Get binary `CORRECT` / `WRONG` plus a one-sentence reasoning.

4. **Score** — Per category and overall: J score (mean of judge), F1 (token overlap with gold), BLEU-1. Plus token and latency means.

5. **Aggregate over N runs** — Mean ± standard deviation per category. Default `N = 3`; support `--runs 10` for a publishable run.

---

## Canonical prompts (use exactly these)

These are taken from the Mem0 evaluation harness conventions. **Do not invent your own** — judge prompt drift is the #1 source of cross-paper disagreement.

### Answer prompt (system)

```
You are an intelligent memory assistant tasked with retrieving accurate
information from conversation memories.

Below you will be shown context retrieved from a long conversation between
two speakers. The current date is {today}. Use ONLY the context to answer
the question. Be concise — give the shortest answer that fully addresses
the question, ideally a span lifted from the context. If the context does
not contain the answer, reply "No information available."

Context:
{retrieved_context}

Question:
{question}

Answer:
```

`{today}` should be set to the date of the latest session in the conversation, so temporal questions resolve correctly. **This is critical for category 3 (temporal) accuracy** — without an anchor date, the LLM has no reference point.

### Judge prompt (system)

```
Your task is to determine whether the AI's generated answer is semantically
equivalent to the gold answer for the given question. Even if phrasing,
formatting, or surface form differs, treat the answer as CORRECT if it
conveys the same factual content. For dates, "May 7th" and "7 May" are
the same. For names, partial names that uniquely identify the referent
count as CORRECT. If the AI says "No information available" but the gold
answer is a real value, that is WRONG.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First give a one-sentence explanation of your reasoning, then on a new
line output exactly one of: CORRECT or WRONG. Do not include both labels.

Return JSON: {"reasoning": "...", "label": "CORRECT" or "WRONG"}
```

Judge model: `gpt-4o-mini` by default (most-published). Allow override via `--judge-model`. Temperature 0.0 for both answer and judge models.

---

## Memory adapter contract

Every adapter under `src/adapters/` implements:

```python
class MemoryAdapter(ABC):
    name: str

    def reset(self, conversation_id: str) -> None:
        """Clear all memory for a fresh conversation."""

    def add_session(
        self,
        conversation_id: str,
        session_id: str,
        timestamp: str,
        turns: list[dict],   # each turn has speaker, text, dia_id
    ) -> AddStats:
        """Ingest one session. Return token + latency stats."""

    def search(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 30,
    ) -> SearchResult:
        """Return retrieved context as a string + metadata
        (retrieved tokens, latency, retrieved chunk ids)."""
```

Three adapters to implement first:

1. **`my_system.py`** — calls into my memory system. Map my system's API to this contract.
2. **`full_context.py`** — `add_session` just appends turns to an in-memory list keyed by conversation_id; `search` returns the entire concatenated conversation. **This is the baseline that beats most memory systems on LoCoMo — if my system can't beat it, that's important to know.**
3. **`naive_rag.py`** — chunk all ingested turns into ~500-token chunks, embed with `text-embedding-3-small`, store in a local FAISS or chroma index, return top-k=30 chunks at search time.

---

## CLI

```bash
# main eval
python scripts/run_full_eval.py \
    --adapter my_system \
    --answer-model gpt-4o-mini \
    --judge-model gpt-4o-mini \
    --runs 3 \
    --output-dir results/my_system_$(date +%Y%m%d) \
    --top-k 30

# baselines
python scripts/run_full_eval.py --adapter full_context --runs 3
python scripts/run_full_eval.py --adapter naive_rag --runs 3 --top-k 30

# dataset sanity check
python scripts/verify_dataset.py --path dataset/locomo10.json
# Should print: 10 conversations, 1986 total QA, 1540 after filtering cat 5
```

---

## Output format

### `results/<run_name>/run_<i>/predictions.json`

```jsonc
[
  {
    "conversation_id": "conv-001",
    "qa_id": "conv-001-q42",
    "category": 2,
    "question": "...",
    "gold_answer": "...",
    "predicted_answer": "...",
    "retrieved_context": "...",
    "stats": {
      "retrieval_tokens": 1234,
      "prompt_tokens": 1500,
      "completion_tokens": 23,
      "search_latency_ms": 142,
      "generation_latency_ms": 890
    },
    "judge": {
      "label": "CORRECT",
      "reasoning": "...",
      "judge_model": "gpt-4o-mini"
    },
    "f1": 0.87,
    "bleu1": 0.62
  }
]
```

### `results/<run_name>/aggregate.json`

```jsonc
{
  "adapter": "my_system",
  "runs": 3,
  "answer_model": "gpt-4o-mini",
  "judge_model": "gpt-4o-mini",
  "filter": "categories_1_to_4",
  "n_questions": 1540,
  "per_category": {
    "1_single_hop":  {"j_mean": 0.72, "j_std": 0.01, "f1_mean": 0.55, "bleu1_mean": 0.41, "n": 838},
    "2_multi_hop":   { ... },
    "3_temporal":    { ... },
    "4_open_domain": { ... }
  },
  "overall": {"j_mean": 0.68, "j_std": 0.01, "f1_mean": 0.51, "bleu1_mean": 0.38},
  "tokens": {"avg_retrieval": 4200, "avg_prompt": 4500, "avg_completion": 22},
  "latency": {"avg_search_ms": 130, "avg_generation_ms": 850}
}
```

### `results/<run_name>/report.md`

Auto-generated comparison table: per-category J score with mean ± std for each adapter, plus a "vs full-context" delta column. This is what gets pasted into a paper or blog post.

---

## Implementation order

Build in this order — don't skip ahead. Each step should be working and tested before the next.

1. **Skeleton + dataset loader + sanity check** (`src/data.py`, `scripts/verify_dataset.py`). Verify category counts: should be ~1,540 after filtering cat 5. **Halt and ask if counts don't match.**
2. **Metrics + tests** (`src/metrics.py`, `tests/test_metrics.py`). F1 and BLEU-1 with golden test cases. Token-level F1 normalized for case/punctuation/articles (SQuAD style).
3. **Judge** (`src/judge.py`, `tests/test_judge.py`). Test against fixtures: paraphrase → CORRECT, wrong fact → WRONG, refusal-when-answer-exists → WRONG. Use gpt-4o-mini, temp=0, structured JSON output.
4. **Full-context adapter** (`src/adapters/full_context.py`). Simplest possible. End-to-end on **one conversation** to validate the whole pipeline before scaling up.
5. **Naive RAG adapter** (`src/adapters/naive_rag.py`). Local FAISS, text-embedding-3-small, chunk size 500, top-k 30.
6. **My system adapter** (`src/adapters/my_system.py`). Stub it first with the three required methods, confirm the harness drives it correctly with a fake in-memory backend, then wire to the real system.
7. **Run experiments** (`scripts/run_full_eval.py`). Single-run on all 10 conversations. Then 3-run for stats.
8. **Reporting** (`src/reporting/`). Generate `aggregate.json` and `report.md`.

---

## Critical gotchas (these will bite if ignored)

1. **Do NOT bulk-load the conversation.** Ingest session-by-session in chronological order. This is the entire point of the benchmark.
2. **Do NOT include adversarial (category 5) in main numbers.** Standard convention. If you want to report cat 5, do it as a separate subsection.
3. **Set `today` in the answer prompt** to the date of the latest session, so temporal questions resolve. Without this, category 3 scores will be artificially low.
4. **Use temperature 0** for both answer and judge models. Non-zero temperature inflates run-to-run variance and makes 3 runs insufficient.
5. **Reset memory between conversations.** `conversation_id` should fully scope memory state. Cross-conversation contamination is a silent killer.
6. **Track tokens honestly.** Retrieval tokens = tokens in retrieved context. Prompt tokens = retrieval + question + system prompt. Some published numbers conflate these and look better than they are.
7. **Multiple runs matter.** A single run can have ±2-3% noise from the judge alone. Default 3 runs; do 10 before publishing anything.
8. **Always include the full-context baseline.** If my system loses to full-context, that is the most important finding in the report. Don't bury it.
9. **Image-dependent questions.** A handful of QA items reference images. Three options, in order of preference: (a) substitute the BLIP caption when the original image is referenced in the relevant turn, (b) skip them with a documented exclusion list, (c) just live with them being hard. Document whichever you choose.
10. **Judge model determines absolute scores.** A run judged by gpt-4o-mini is not directly comparable to one judged by gpt-4.1. State the judge model prominently in every report.

---

## Dependencies

```toml
[project]
dependencies = [
  "openai>=1.40",
  "tiktoken>=0.7",
  "numpy",
  "pandas",
  "tqdm",
  "python-dotenv",
  "pydantic>=2",
  "faiss-cpu",            # for naive_rag
  "nltk",                 # BLEU
  "rich",                 # nicer CLI output
]
[dependency-groups]
dev = ["pytest", "pytest-asyncio", "ruff"]
```

Use `uv` for env management. Python 3.11+.

---

## Definition of done

- [ ] `verify_dataset.py` reports 10 conversations, 1986 total QA, 1540 after cat-5 filter
- [ ] All tests in `tests/` pass
- [ ] `full_context` baseline runs end-to-end on all 10 conversations
- [ ] `naive_rag` baseline runs end-to-end on all 10 conversations
- [ ] `my_system` adapter runs end-to-end on all 10 conversations
- [ ] Three full runs of each adapter complete, results saved
- [ ] `report.md` generated with per-category J score (mean ± std), F1, BLEU-1, tokens, latency for all three adapters side by side
- [ ] README documents how to reproduce, including dataset download step

---

## Stretch (do not start until the above is done)

- **LongMemEval** pipeline using the same harness — `https://github.com/xiaowu0162/LongMemEval`. Same adapter contract, different dataset and prompts. Becoming the second standard.
- **Multiple judge models** — run the same predictions through gpt-4o-mini AND gpt-4.1 judges, report both. Demonstrates judge-robustness.
- **Confidence intervals** — Wilson score CIs per category (the locomo-audit repo notes that some categories are too small for tight CIs; report this honestly).

---

## Questions to surface before coding

If anything below is unclear, **ask before writing code**:

1. What's the API surface of my memory system? (Will determine `my_system.py` adapter shape.)
2. Are we running OpenAI directly or through a proxy?
3. Self-hosted embeddings or OpenAI embeddings for the RAG baseline?
4. Budget cap? A 3-run eval on 1,540 questions with gpt-4o-mini answer + judge is roughly $5–15 depending on retrieved context size. Full-context baseline costs more.
