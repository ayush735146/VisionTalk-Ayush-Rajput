"""Train VisionTalk on Flickr8k.
BY AYUSH RAJPUT
"""
from pathlib import Path
import argparse, json, pickle
import numpy as np
from tensorflow.keras.utils import Sequence
from preprocess import prepare_metadata, create_sequences
from model import build_merge_model

class CaptionSequence(Sequence):
    def __init__(self, samples, features, batch_size=64, shuffle=True):
        self.samples = samples
        self.features = features
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(samples))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = [self.samples[i] for i in batch_idx]
        X_img = np.stack([self.features[s[0]] for s in batch]).astype("float32")
        X_txt = np.stack([s[1] for s in batch]).astype("int32")
        y = np.array([s[2] for s in batch], dtype="int32")
        return {"image_features": X_img, "caption_sequence": X_txt}, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def build_samples(image_ids, captions, tokenizer, max_length, features):
    samples = []
    for image_id in image_ids:
        if image_id not in features or image_id not in captions:
            continue
        for caption in captions[image_id]:
            for item in create_sequences(image_id, caption, tokenizer, max_length):
                samples.append(item)
    return samples

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_images", type=int, default=0)
    args = p.parse_args()

    root = Path("data")
    text = root / "Flickr8k_text"
    captions, tokenizer, max_length, train_ids, dev_ids, test_ids = prepare_metadata(
        text/"Flickr8k.token.txt", text/"Flickr_8k.trainImages.txt",
        text/"Flickr_8k.devImages.txt", text/"Flickr_8k.testImages.txt"
    )
    with open("artifacts/features.pkl", "rb") as f:
        features = pickle.load(f)

    if args.max_images:
        train_ids = train_ids[:args.max_images]
        dev_ids = dev_ids[:max(50, args.max_images//10)]

    train_samples = build_samples(train_ids, captions, tokenizer, max_length, features)
    dev_samples = build_samples(dev_ids, captions, tokenizer, max_length, features)

    vocab_size = min(len(tokenizer.word_index) + 1, tokenizer.num_words or len(tokenizer.word_index)+1)
    model = build_merge_model(vocab_size, max_length)
    model.summary()

    history = model.fit(
        CaptionSequence(train_samples, features, args.batch_size),
        validation_data=CaptionSequence(dev_samples, features, args.batch_size, shuffle=False),
        epochs=args.epochs
    )
    Path("artifacts").mkdir(exist_ok=True)
    model.save("artifacts/caption_model.keras")
    Path("artifacts/training_history.json").write_text(
        json.dumps({k:[float(x) for x in v] for k,v in history.history.items()}, indent=2),
        encoding="utf-8"
    )
    Path("artifacts/test_captions.json").write_text(
        json.dumps({k: captions[k] for k in test_ids if k in captions}, indent=2),
        encoding="utf-8"
    )
    print("Training complete.")

if __name__ == "__main__":
    main()
