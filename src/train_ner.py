from pathlib import Path

import spacy
from spacy.cli.init_config import init_config
from spacy.cli.train import train as spacy_train
from spacy.util import Config


def train_custom_ner(train_docbin: Path, dev_docbin: Path, output_dir: Path, epochs: int = 20):
    config = init_config(lang="en", pipeline=["ner"], optimize="efficiency")
    config["paths"]["train"] = str(train_docbin)
    config["paths"]["dev"] = str(dev_docbin)
    config["training"]["max_epochs"] = epochs
    config["training"]["patience"] = 10 ** 9

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = output_dir / "ner_config.cfg"
    cfg_path.write_text(Config(config).to_str())

    spacy_train(str(cfg_path), str(output_dir))
    return spacy.load(output_dir / "model-best")
