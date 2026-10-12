import random
from pathlib import Path

TRAIN_COMPANIES = [
    "ACME Logistics N.V.",
    "Global Parts Distribution GmbH",
    "Meridian Industrial Supply SARL",
    "Northwind Components Ltd",
    "Aurora Freight Systems BV",
    "Cascade Manufacturing SA",
    "Delta Precision Tools AB",
    "Falcon Heavy Industries SpA",
    "Harbor Contracting Oy",
    "Zenith Energy Trading PLC",
]
NOVEL_COMPANIES = [
    "Sentinel Robotics Ltd",
    "Vantage Chemical Group SpA",
    "Pinnacle Textile Works SA",
    "Orbital Packaging Systems AB",
    "Trail Wind Turbines Oy",
]
BRANCHES = ["Rotterdam", "Antwerp-Brussels", "Munich", "Lyon", "Groningen", "Porto", "Krakow", "Bergen"]


def random_invoice_text(rng: random.Random, companies):
    company = rng.choice(companies)
    inv = weighted(rng, [
        (f"INV-{rng.randint(2023, 2027)}-{rng.randint(1000, 9999)}", 0.6),
        (f"BIL-{rng.randint(2023, 2027)}-{rng.randint(100000, 999999)}", 0.4),
    ])
    vat = weighted(rng, [
        (f"BE{rng.randint(10**9, 10**10 - 1)}", 0.5),
        (f"FR{rng.randint(10**9, 10**10 - 1)}", 0.3),
        (f"DE{rng.randint(10**9, 10**10 - 1)}", 0.2),
    ])
    year, month, day = rng.randint(2023, 2027), rng.randint(1, 12), rng.randint(1, 28)
    date = weighted(rng, [
        (f"{year:04d}-{month:02d}-{day:02d}", 0.5),
        (f"{day:02d}/{month:02d}/{year:04d}", 0.3),
        (f"{month:02d} {day:02d} {year}", 0.2),
    ])
    amount = f"{rng.randint(50, 950000) / 100:,.2f}"
    header_style = rng.choice(["INVOICE", "FACTURE - INVOICE", "Tax Invoice"])
    vat_key = rng.choice(["VAT No:", "VATIN:", "Tax ID:"])
    total_key = rng.choice(["Total Amount:", "TOTAL DUE:", "Amount Payable:"])
    footer = rng.choice(["Payment terms: 30 days.\n", "Please reference the invoice number.\n", f"Branch: {rng.choice(BRANCHES)}\n"])
    segments = [
        (None, header_style + "\n"),
        (None, "From: "),
        ("COMPANY", company),
        (None, "\nInvoice No: "),
        ("INV_NUM", inv),
        (None, "\nIssue Date: "),
        ("DATE", date),
        (None, f"\n{vat_key} "),
        ("VAT_NUM", vat),
        (None, f"\n{total_key} "),
        ("AMOUNT", f"EUR {amount}" if rng.random() < 0.7 else f"{rng.choice(['USD', 'CHF'])} {amount}"),
        (None, "\n" + footer),
    ]
    text = ""
    entities = []
    for label, part in segments:
        if label:
            entities.append((len(text), len(text) + len(part), label))
        text += part
    return text, entities


def weighted(rng: random.Random, pairs):
    r = rng.random()
    acc = 0.0
    for val, w in pairs:
        acc += w
        if r < acc:
            return val
    return pairs[-1][0]


def generate_split(n: int, seed: int, companies=None):
    rng = random.Random(seed)
    texts, ents = [], []
    for _ in range(n):
        t, e = random_invoice_text(rng, companies or TRAIN_COMPANIES)
        texts.append(t)
        ents.append(e)
    return texts, ents


def build_docbin(texts, ents, out_path: Path):
    import spacy
    from spacy.tokens import DocBin
    from spacy.training import Example

    nlp = spacy.blank("en")
    db = DocBin()
    for text, offsets in zip(texts, ents):
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, {"entities": offsets})
        db.add(example.reference)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    db.to_disk(out_path)
    return out_path
