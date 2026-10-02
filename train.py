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
