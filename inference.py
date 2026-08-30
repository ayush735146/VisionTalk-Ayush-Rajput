"""Greedy inference for VisionTalk.
BY AYUSH RAJPUT
"""
from pathlib import Path
import json, pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

def load_artifacts():
    model = load_model("artifacts/caption_model.keras")
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    max_length = json.loads(Path("artifacts/max_length.json").read_text())["max_length"]
    return model, tokenizer, max_length

def image_feature(image_path):
    encoder = VGG16(weights="imagenet", include_top=False, pooling="max")
    img = Image.open(image_path).convert("RGB").resize((224,224))
    arr = preprocess_input(np.expand_dims(np.asarray(img, dtype=np.float32), 0))
    return encoder.predict(arr, verbose=0)[0]

def generate_caption(model, tokenizer, max_length, feature, max_words=40):
    reverse = {v:k for k,v in tokenizer.word_index.items()}
    sequence = ["startseq"]
    for _ in range(max_words):
        encoded = tokenizer.texts_to_sequences([" ".join(sequence)])[0]
        padded = np.zeros((1, max_length), dtype=np.int32)
        padded[0, :min(len(encoded), max_length)] = encoded[:max_length]
        probs = model.predict(
            {"image_features": feature.reshape(1,-1), "caption_sequence": padded},
            verbose=0
        )[0]
        idx = int(np.argmax(probs))
        word = reverse.get(idx, "<unk>")
        if word == "endseq":
            break
        if word not in {"startseq", "<unk>"}:
            sequence.append(word)
    return " ".join(sequence[1:])

def caption_image(image_path):
    model, tokenizer, max_length = load_artifacts()
    feature = image_feature(image_path)
    return generate_caption(model, tokenizer, max_length, feature)
