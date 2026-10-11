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
- **Do:** rerun smoke varying `--epochs` 3→12; watch loss + eval score interplay.

## L8. The comparison that matters
- **Understand:** our EntityRuler = perfect precision, recall dropped on DATE variant + NOVEL companies (test-only names). NER recovered everything *learned weights generalize from context*, not identity lists.
- **Generate:** from `sample_misses_trained_ner`, categorize failures and argue in 6 sentences which production entity classes should stay rule vs become model.

# Phase 3 — Hybrid + engineering

## L9. Hybrid merge semantics
- **Understand:** `merge_hybrid` = NER wins overlaps, ruler adds non-overlapping spans (its specialty: rare deterministic formats).
- **Do:** verify hybrid ≥ max(NER, ruler) on F1 in the full Kaggle run.
- **Generate:** what failure mode could make the hybrid WORSE than pure NER? Name one and design a pre-merge filter.

## L10. Production ships
- **Do:** full Kaggle run (320 train / 60 dev / 120 test, 24 epochs, no GPU); pull `metrics.json`.
- **L11.** FastAPI `POST /invoice` returning JSON with entities + confidence flags, Dockerfile, tests for rules (deterministic) and NER (statistical).
- **L12.** Then port to CoNLL-2003 as second corpus (real data, messier): attach a Kaggle CoNLL dataset, retrain, and compare micro-F1 with the synthetic invoices — write down honestly what the synthetic gap means.

---

# Flashcards (spaced review — never explore with these in front of you)

- C1 (L1): The 3 components of an entity annotation, and where offsets are anchored (start_char inclusive, end_char exclusive).
- C2 (L2): How spaCy tokenizes `INV-2025-5012`, `12,345.67`, and `2025-04-23`.
- C3 (L3): What object stores training data on disk, and which function raises on misaligned offsets?
- C4 (L4): The 5 pattern predicates in the token DSL used by the baseline.
- C5 (L5): Why does the COMPANY EntityRuler have recall 0.667 on our test set but still enough to be useful? trade-off.
- C6 (L6): What component lives before NER inside the pipeline, and what does it do?
- C7 (L7): What does `patience` mean in spaCy 3.7+? What mistake did we hit and how was it caught?
- C8 (L8): Lessons on when rules generalize better than a model (rare deterministic formats) and vice versa (context-based identity patterns).
- C9 (L9): Hybrid merge operator — give the 1-sentence contract and one way it breaks.
- C10 (L10): Which of the 3 systems believed in window-dressing and which didn't: sample the misses list to argue.

# Spacing protocol
- Session opens: 4 random cards cold.
- D+2 after L6: patch one rule pattern live, compare against the trained model (5 min).
- D+7 after L8: re-derive by hand how Example.from_dict enforces alignment (whiteboard & error path).
- D+21: rebuild the full pipeline from scratch, no notes.

# Final gate (portfolio checklist)
EDA → baseline → implement → experiment → evaluate → deploy → containerize → monitor → document, with ruler-vs-NER-vs-hybrid table front and center.
