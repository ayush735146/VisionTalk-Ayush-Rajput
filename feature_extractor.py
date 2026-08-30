"""VGG16 feature extraction for VisionTalk.
BY AYUSH RAJPUT
"""
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

DATASET_DIR = Path("data/Flickr8k_Dataset")
FEATURES_FILE = Path("artifacts/features.pkl")

def build_encoder():
    # VGG16's classifier is removed; the flattened convolutional representation
    # is a 4096-dimensional vector.
    return VGG16(weights="imagenet", include_top=False, pooling="max")

def extract_features(image_dir=DATASET_DIR, output_file=FEATURES_FILE):
    image_dir = Path(image_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    model = build_encoder()
    features = {}
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")))

    for i, path in enumerate(image_paths, 1):
        try:
            image = Image.open(path).convert("RGB").resize((224, 224))
            arr = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
            arr = preprocess_input(arr)
            vector = model.predict(arr, verbose=0)[0].astype(np.float32)
            features[path.name] = vector
            if i % 100 == 0:
                print(f"Processed {i}/{len(image_paths)} images")
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    with open(output_file, "wb") as f:
        pickle.dump(features, f)
    print(f"Saved {len(features)} feature vectors to {output_file}")
    return features

if __name__ == "__main__":
    extract_features()
