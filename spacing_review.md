# Spaced review — flashcards D1–D10

Reviewing days (cards covered 2026-11-07):

- **D+2 → 2026-11-09**
- **D+7 → 2026-11-14**
- **D+21 → 2026-11-28**

## Cards

- D1 (L2): Why generate synthetic invoices online (draw + span offsets) instead of annotating real PDFs?
- D2 (L3): Why record `(start, end, label)` offsets in the generator's return, not the label strings?
- D3 (L4): `merge_hybrid` — why must NER span spans WIN over ruler spans, not merge outside them?
- D4 (L5): Why does the EntityRuler run *before* the trained NER in the hybrid merge? (Which failure mode would the reverse order create?)
- D5 (L6): Why was the training config's `patience` set to 10^9 instead of `None`?
- D6 (L6): Name the two invoice-number prefixes the EntityRuler registers (`INV-`/`BIL-`) and where the patterns live in `src/rules.py`.
- D7 (L9): Why does dev F1=1.000 stay flat across NER seeds while the novel-company set swings 0.86–0.93, and what does that say about trusting synthetic dev metrics?
- D8 (L9): When a NER entity overlaps a ruler entity, which one survives in `merge_hybrid` — and why was that order chosen?
- D9 (L11): In `src/api.py`, why is a degenerate 1-char entity (e.g. `"a"` as DATE) filtered in the API layer rather than patched into `evaluation.merge_hybrid`?
- D10 (L12): The API's `/health` endpoint names the model path — what artifacts does `checkpoints/ner_model_final` bundle and with which seed and doc count was it locally retrained?

## When you can't recall

- Never with the answer in front of you.
- If a card stalls at D+2 AND D+7, it moves to the "stock-pile" and is re-generated as a new question at D+21.
- Truth check: `grep -rn "merge_hybrid\|required" src/` only AFTER writing your answer down.
