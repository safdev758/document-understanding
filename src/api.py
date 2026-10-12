import os
from contextlib import asynccontextmanager
from pathlib import Path

import spacy
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .evaluation import extract_ents, merge_hybrid
from .rules import build_ruler_pipeline

BUNDLE_ROOT = Path(os.environ.get("BUNDLE_ROOT", Path(__file__).resolve().parent.parent))
MODEL_PATH = Path(os.environ.get("NER_MODEL", BUNDLE_ROOT / "checkpoints" / "ner_model_final"))
REQUIRED_LABELS = ("COMPANY", "INV_NUM", "DATE", "VAT_NUM", "AMOUNT")


class ExtractResponse(BaseModel):
    entities: list[dict]
    missing: list[str]
    suspicious: bool = Field(..., description="True when a required field is missing")


def create_app():
    state = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ner = spacy.load(MODEL_PATH)
        ruler = build_ruler_pipeline([])
        state.update(ner=ner, ruler=ruler)
        yield

    app = FastAPI(title="Invoice Information Extraction API", version="1.0", lifespan=lifespan)

    def extract(text: str) -> ExtractResponse:
        ner_ents = extract_ents(state["ner"], [text])[0]
        ruler_ents = extract_ents(state["ruler"], [text])[0]
        merged = merge_hybrid([ner_ents], [ruler_ents])[0]
        entities = [
            {"start": s, "end": e, "label": l, "text": text[s:e]}
            for s, e, l in merged
            if len(text[s:e].strip()) >= 2
        ]
        found = {item["label"] for item in entities}
        missing = [lab for lab in REQUIRED_LABELS if lab not in found]
        return ExtractResponse(entities=sorted(entities, key=lambda i: i["start"]), missing=missing, suspicious=bool(missing))

    @app.get("/health")
    def health():
        return {"status": "ok", "model": str(MODEL_PATH), "required": list(REQUIRED_LABELS)}

    @app.post("/invoice", response_model=ExtractResponse)
    def invoice_json(req: dict):
        text = req.get("text")
        if not text or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="body must be JSON with a \"text\" field")
        return extract(text)

    @app.post("/invoice_upload", response_model=ExtractResponse)
    def invoice_file(file: UploadFile = File(...)):
        try:
            text = file.file.read().decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="file must be UTF-8 text")
        return extract(text)

    return app


app = create_app()
