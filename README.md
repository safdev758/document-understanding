# 8. Intelligent Document Understanding & Information Extraction

**Priority in portfolio:** #3 — spaCy, NER, information extraction
**Difficulty:** 9/10 · **Domain:** spaCy (NLP)

A document intelligence system for unstructured documents.

## What it does

- Receives CVs, invoices, contracts and business documents.
- Extracts names, organizations, dates and monetary values.
- Identifies relevant entities.
- Categorizes documents.
- Outputs structured information as JSON.
- Flags missing or suspicious information.

## Datasets

- **CoNLL-2003** (general NER): https://www.tensorflow.org/datasets/catalog/conll2003
- **FUNSD** (form understanding)
- **CORD** (receipt understanding)

## Models and techniques

- spaCy NER
- Custom entity recognition
- EntityRuler
- Dependency parsing
- Custom text categorization
- spaCy training pipelines https://spacy.io/usage/training
  (spaCy uses CNN-based / statistical / transformer architectures internally —
  spaCy is a framework, not a single neural architecture)

## What makes it impressive

A **custom NER model** for company names, invoice numbers, VAT numbers, dates,
monetary values and customer names — exposed through an API.

> See the master document `../../ai-projects-portfolio.md` for the shared
> system architecture, the 9 mandatory ML-engineering requirements, and MLflow
> experiment tracking guidance.

---

### How to Run Your Own Training on Kaggle

#### Option 1: Standard Kaggle CLI

1. Inside your project folder, create `kernel-metadata.json`:

    ```json
    {
      "id": "khentitsafouaneamine/document-intelligence-train",
      "title": "document intelligence train",
      "code_file": "train.py",
      "language": "python",
      "kernel_type": "script",
      "is_private": true,
      "enable_gpu": true,
      "enable_internet": true
    }
    ```

2. Push and launch training:

    ```bash
    kaggle kernels push -p .
    ```

3. Check status:

    ```bash
    kaggle kernels status khentitsafouaneamine/document-intelligence-train
    ```

4. Download trained checkpoints / outputs:

    ```bash
    kaggle kernels output khentitsafouaneamine/document-intelligence-train -p ./output
    ```

#### Option 2: The Multi-File Project Bundler (`kaggle_runner.py`)

```bash
# Bundle your whole project, enable GPU, watch progress, and auto-download results:
python3 kaggle_runner.py bundle --src ./08-document-understanding \
    --entry train.py --slug document-intelligence-train --watch
```

> Run the bundle command from `projects/` (the parent directory), pointing
> `--src` at this folder.

### Production serving (L13): API + Docker + MLflow

The serving stack bundles the locally retrained NER model
(`checkpoints/ner_model_final`, same seed/params as the Kaggle kernel: 320 train
