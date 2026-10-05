import spacy
from spacy.pipeline import EntityRuler


def add_invoice_ruler(nlp, known_companies):
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "ORTH"})
    ruler.add_patterns([
        {"label": "INV_NUM", "pattern": [{"TEXT": {"REGEX": "^(INV|BIL)-\\d{4}$"}}, {"ORTH": "-"}, {"LIKE_NUM": True}]},
        {"label": "INV_NUM", "pattern": [{"LOWER": "invoice", "OP": "+"}, {"ORTH": "/", "OP": "?"}, {"LIKE_NUM": True}, {"ORTH": "/", "OP": "?"}, {"LIKE_NUM": True}]},
        {"label": "VAT_NUM", "pattern": [{"TEXT": {"REGEX": "^[A-Z]{2}\\d{9,10}$"}}]},
        {"label": "DATE", "pattern": [{"LIKE_NUM": True}, {"ORTH": "-"}, {"LIKE_NUM": True}, {"ORTH": "-"}, {"LIKE_NUM": True}]},
        {"label": "DATE", "pattern": [{"LIKE_NUM": True}, {"ORTH": "/"}, {"LIKE_NUM": True}, {"ORTH": "/"}, {"LIKE_NUM": True}]},
        {"label": "DATE", "pattern": [{"LIKE_NUM": True}, {"LIKE_NUM": True}, {"LIKE_NUM": True}]},
        {"label": "AMOUNT", "pattern": [{"ORTH": {"IN": ["EUR", "$", "€", "USD", "CHF"]}}, {"LIKE_NUM": True}]},
    ] + [{"label": "COMPANY", "pattern": c} for c in known_companies])
