"""Caption cleaning, vocabulary and sequence generation.
BY AYUSH RAJPUT
"""
from pathlib import Path
import re, json, pickle
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

START = "startseq"
END = "endseq"

def clean_caption(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_captions(token_file):
    mapping = {}
    with open(token_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            image_id, caption = parts
            image_name = image_id.split("#")[0]
            cleaned = clean_caption(caption)
            mapping.setdefault(image_name, []).append(f"{START} {cleaned} {END}")
    return mapping

def load_split(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        return [Path(line.strip()).name for line in f if line.strip()]

def build_tokenizer(captions, vocab_size=None):
    texts = [c for caps in captions.values() for c in caps]
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>", filters="")
    tokenizer.fit_on_texts(texts)
    return tokenizer

def create_sequences(image_id, caption, tokenizer, max_length):
    seq = tokenizer.texts_to_sequences([caption])[0]
    for i in range(1, len(seq)):
        in_seq = pad_sequences([seq[:i]], maxlen=max_length, padding="post")[0]
        out_word = seq[i]
        yield image_id, in_seq, out_word

def prepare_metadata(caption_file, train_file, dev_file, test_file, artifact_dir="artifacts"):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    captions = load_captions(caption_file)
    train_ids, dev_ids, test_ids = map(load_split, [train_file, dev_file, test_file])

    train_caps = {k: captions[k] for k in train_ids if k in captions}
    tokenizer = build_tokenizer(train_caps)
    max_length = max(len(tokenizer.texts_to_sequences([c])[0]) for caps in captions.values() for c in caps)

    with open(artifact_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    (artifact_dir / "max_length.json").write_text(json.dumps({"max_length": max_length}), encoding="utf-8")
    (artifact_dir / "splits.json").write_text(
        json.dumps({"train": train_ids, "dev": dev_ids, "test": test_ids}), encoding="utf-8"
    )
    (artifact_dir / "captions.json").write_text(json.dumps(captions), encoding="utf-8")
    return captions, tokenizer, max_length, train_ids, dev_ids, test_ids
