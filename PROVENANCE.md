# Provenance — every artifact traces to a real run

## Kaggle kernels (account khentitsafouaneamine)

| kernel | purpose | key outputs |
| --- | --- | --- |
| `document-intelligence-train` | spaCy 3.8 custom NER on synthetic invoices (320 train docs, seed 2025, 24 epochs) | dev 1.000 P/R, kernel-logged novel + hybrid metrics |

## Local re-training (identical seed + params to kernel)

- `checkpoints/ner_model_final` — rebuilt locally with the same generator seed
  2025, 320 docs, efficiency config, 24 epochs so the API ships without a kernel
  dependency ("provenance" for `checkpoints/ner_model_final` regenerates from
  `spacy.cli.train` via the local seed config)
- `output/hybrid_local_metrics.json` — dev 1.000, novel-company 0.93 P/R
- `output/seed_local_metrics.json` — 3-seed robustness study (dev F1 flat 1.000,
  novel F1 0.932 / 0.860 / 0.932)
