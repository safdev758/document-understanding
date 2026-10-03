import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.evaluation import evaluate_ents, extract_ents, merge_hybrid
from src.invoices import NOVEL_COMPANIES, TRAIN_COMPANIES, build_docbin, generate_split
from src.rules import build_ruler_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Document Intelligence — EntityRuler vs trained NER vs hybrid")
    p.add_argument("--n-train", type=int, default=160)
    p.add_argument("--n-dev", type=int, default=40)
    p.add_argument("--n-test", type=int, default=60)
    p.add_argument("--novel-ratio", type=float, default=0.6, help="fraction of TEST docs whose company is never seen in training")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    workdir = Path("work")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("[DATA] generating synthetic invoices...")
    train_texts, train_ents = generate_split(args.n_train, args.seed, TRAIN_COMPANIES)
    dev_texts, dev_ents = generate_split(args.n_dev, args.seed + 1, TRAIN_COMPANIES)
    test_texts, test_ents = generate_split(args.n_test, args.seed + 2, TRAIN_COMPANIES)
    n_novel = int(args.n_test * args.novel_ratio)
    novel_texts, novel_ents = generate_split(n_novel, args.seed + 3, NOVEL_COMPANIES)
    for t, e in zip(novel_texts, novel_ents):
        test_texts.append(t)
        test_ents.append(e)
    print(f"[DATA] train={len(train_texts)} dev={len(dev_texts)} test={len(test_texts)} ({n_novel} with NOVEL company names)")

    train_db = build_docbin(train_texts, train_ents, workdir / "train.spacy")
    build_docbin(dev_texts, dev_ents, workdir / "dev.spacy")

    metrics = {"meta": vars(args)}

    print("\n[BASELINE] EntityRuler-only...")
    ruler_nlp = build_ruler_pipeline(TRAIN_COMPANIES)
    baseline = evaluate_ents(extract_ents(ruler_nlp, test_texts), test_ents)
    metrics["entity_ruler"] = baseline
    print("[RULER]", json.dumps(baseline["micro"]), json.dumps(baseline["per_type"]))

    print(f"\n[NER] training custom spaCy NER ({args.epochs} epochs)...")
    from src.train_ner import train_custom_ner
    ner_nlp = train_custom_ner(train_db, workdir / "dev.spacy", workdir / "ner_model", epochs=args.epochs)
    ner_nlp.to_disk(workdir / "ner_model_final")
    ner_result = evaluate_ents(extract_ents(ner_nlp, test_texts), test_ents)
    metrics["trained_ner"] = ner_result
    print("[NER]", json.dumps(ner_result["micro"]), json.dumps(ner_result["per_type"]))

    print("\n[HYBRID] NER + EntityRuler gap-filling...")
    hybrid = evaluate_ents(
        merge_hybrid(extract_ents(ner_nlp, test_texts), extract_ents(ruler_nlp, test_texts)),
        test_ents,
    )
    metrics["hybrid"] = hybrid
    print("[HYBRID]", json.dumps(hybrid["micro"]), json.dumps(hybrid["per_type"]))
