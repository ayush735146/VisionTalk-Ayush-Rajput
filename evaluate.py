"""BLEU-1/BLEU-4 evaluation for VisionTalk.
BY AYUSH RAJPUT
"""
from pathlib import Path
import json, pickle, re
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from inference import load_artifacts, image_feature, generate_caption

def tokenize(sentence):
    return re.findall(r"[a-z]+", sentence.lower())

def main():
    test = json.loads(Path("artifacts/test_captions.json").read_text())
    model, tokenizer, max_length = load_artifacts()
    refs, hyps, examples = [], [], []
    smooth = SmoothingFunction().method1

    for i, (image_id, captions) in enumerate(test.items()):
        image_path = Path("data/Flickr8k_Dataset") / image_id
        if not image_path.exists():
            continue
        pred = generate_caption(model, tokenizer, max_length, image_feature(image_path))
        refs.append([tokenize(c.replace("startseq ","").replace(" endseq","")) for c in captions])
        hyps.append(tokenize(pred))
        if len(examples) < 10:
            examples.append({"image": image_id, "generated": pred, "actual": captions})
        if i >= 999:
            break

    if not refs:
        raise RuntimeError("No test images found.")
    bleu1 = corpus_bleu(refs, hyps, weights=(1,0,0,0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs, hyps, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
    result = {"BLEU-1": bleu1, "BLEU-4": bleu4, "examples": examples}
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/evaluation_examples.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"BLEU-1": bleu1, "BLEU-4": bleu4}, indent=2))

if __name__ == "__main__":
    main()
