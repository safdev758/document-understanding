# Project 8 — Document Intelligence with spaCy: Full Curriculum (library-based / production)

**Stack:** spaCy 3.8 (EntityRuler, NER training via `spacy train` + TOML configs, `Example`/`DocBin` API) + srsly
**Data:** self-generated synthetic invoices (exact span offsets — the production-safe way to build annotation data)
**Kernels:** `document-intelligence-smoke` → `document-intelligence-train`

---

# Learning Framework (every lesson)

1. **Understand** — first-principles before API. 2. **Do** — run the repo immediately. 3. **Generate** — no-scaffold tasks. 4. **Remember Separately** — flashcards below. 5. **Spacing** — cold-recall gate opens every session; drills at D+2, D+7, D+21.

---

# The repo you will be studying (implementation map)

| File | Role | Lesson |
|---|---|---|
| `src/invoices.py` | synthetic invoice generator: weighted format variants per entity type + exact-offset span bookkeeping + DocBin serialization | L1–L3 |
| `src/rules.py` | EntityRuler baseline: REGEX/LIKE_NUM token patterns for INV_NUM, VAT_NUM, DATE, AMOUNT + phrase patterns for known companies | L4–L5 |
| `src/train_ner.py` | statistical NER arm: `init_config` → TOML config → `spacy.train`, `model-best` loading | L6–L8 |
| `src/evaluation.py` | P/R/F1 micro + per-type, exact-span matching, `merge_hybrid` (NER priority, ruler fills gaps) | L9–L10 |
| `train.py` | orchestrator: data gen → ruler eval → NER training → NER eval → hybrid eval → `metrics.json` (+20 misses sample) | L1 |

Local smoke: `python3 train.py --n-train 40 --n-dev 8 --n-test 12 --novel-ratio 0.5 --epochs 12`

---

# Phase 0 — Data as spans (offsets first)

## L1. Span bookkeeping is THE hard part of annotation
- **Understand:** entities live at (start_char, end_char, LABEL); changing one character earlier shifts every later offset. Our generator builds text left→right appending AFTER recording the offset: zero drift by construction.
- **Do:** read `random_invoice_text`, recompute one entity's offsets by hand.
- **Generate:** add a "Purchase Order" field with your own new segment with correct offsets (test asserts add up).

## L2. Tokenization determines annotation feasibility
- **Understand:** spaCy tokenizes `INV-2025` as ONE token, `2025-04-23` into 5 tokens, `BIL-2023-175954` as 1. You CANNOT annotate inside a token (alignment error).
- **Do:** run the tokenizer dump script; predict then verify für `12,345.67` and `~30 days`.
- **Generate:** write a quick script that takes any list of entity variants and reports the tokens each produces — make this a debugging habit.

## L3. DocBin + Example contract
- **Understand:** `spacy.blank("en")` → `make_doc` → `Example.from_dict(doc, {"entities": offsets})` → validated & added to `DocBin`. Misaligned offsets raise errors: the format forces you to be aligned. (`*DocBin.get_docs()` round-trip.)

# Phase 1 — The rule-based baseline (EntityRuler)

## L4. Token patterns as a DSL
- **Understand:** pattern = sequence of token predicates (ORTH/LOWER/LIKE_NUM/REGEX/OP quantifiers). phrase_matcher_attr used for the phrase-book COMPANY list.
- **Do:** run the baseline; see where it's perfect (P=1.0) and where it starves (recall on variants).
- **Generate:** why does the `{LIKE_NUM, LIKE_NUM, LIKE_NUM}` DATE pattern risk overmatching in arbitrary business text? Give two real-world counter-examples.

## L5. Critical thinking about rules
- **Generate:** € vs USD variant was caught by shifting to a `{"IN": [...]}` — extend to handle `9,355.68 EUR` (post-nominal) without losing precision.

# Phase 2 — The statistical arm (trained NER)

## L6. spaCy programming model
- **Understand:** pipeline = tok2vec + ner; "model-best" chosen by dev ents_f. TOML config; `spacy.cli.train` = SDK path (config-as-code).
- **Do:** open the generated `ner_config.cfg`, locate `max_epochs`, `patience` (spaCy ≤3.7 counts steps; 3.8 counts... check!), `eval_frequency`.

## L7. Training loop economics
- **Understand:** loss versus eval frequency: step-0 eval has loss 0 (nothing learned); we stopped too early once through patience misconfiguration (see git history); with patience=inf we hit F1=1.0 at step 200 on 40 tiny docs — synthetic regularity caveat!
