# VisionTalk — Image Captioning with CNN + LSTM
**BY AYUSH RAJPUT**

A complete academic implementation of the supplied project specification. VisionTalk combines a pretrained CNN encoder with an LSTM decoder to generate natural-language captions for images.

## Project structure
- `app.py` — Streamlit web interface
- `train.py` — end-to-end training entry point
- `feature_extractor.py` — VGG16 feature extraction
- `preprocess.py` — caption cleaning, vocabulary and sequence preparation
- `model.py` — CNN + LSTM merge architecture
- `inference.py` — greedy caption generation
- `evaluate.py` — BLEU-1 / BLEU-4 evaluation and qualitative examples
- `download_flickr8k.py` — dataset download helper
- `requirements.txt` — Python dependencies
- `VisionTalk_Image_Captioning.ipynb` — guided notebook
- `docs/VisionTalk_Theory_Report.pdf` — theory/evaluation report
- `docs/VisionTalk_Presentation.pptx` — final presentation
- `LIVE_LINK.txt` — deployment status/instructions

## 1. Environment
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Dataset
The project specification recommends Flickr8k/Flickr30k with five human-written captions per image.

Place the Flickr8k files like this:

```text
data/
  Flickr8k_Dataset/
    *.jpg
  Flickr8k_text/
    Flickr8k.token.txt
    Flickr_8k.trainImages.txt
    Flickr_8k.devImages.txt
    Flickr_8k.testImages.txt
```

You can also run:

```bash
python download_flickr8k.py
```

The downloader uses the Kaggle API if credentials are configured. Alternatively, manually download Flickr8k and place it in the structure above.

## 3. Extract CNN features
```bash
python feature_extractor.py
```

This uses ImageNet-pretrained VGG16 without the classification head and stores a 4096-dimensional feature vector per image.

## 4. Train
```bash
python train.py --epochs 20
```

For a first test, use fewer epochs:

```bash
python train.py --epochs 1 --max_images 500
```

Training produces:
```text
artifacts/
  tokenizer.pkl
  max_length.json
  caption_model.keras
  test_captions.json
  training_history.json
```

## 5. Evaluate
```bash
python evaluate.py
```

The evaluator reports BLEU-1 and BLEU-4 and saves qualitative examples to `outputs/evaluation_examples.json`.

## 6. Run the web app
```bash
streamlit run app.py
```

Upload an image and VisionTalk will generate a caption.

## Architecture
Image → VGG16 convolutional encoder → 4096-D feature vector → Dense/Dropout

Partial caption → Tokenizer → Embedding → LSTM

Both branches → Concatenate → Dense → Softmax → next word

Training uses teacher forcing and categorical cross-entropy, as specified in the supplied project brief.

## Important note about the live link
A public Streamlit URL cannot be created from this chat because deployment requires access to a hosting account/repository. `LIVE_LINK.txt` contains the exact deployment steps and is intentionally marked as **Pending Deployment** rather than inventing a URL.

## Academic integrity
This package contains the complete source pipeline. Model weights are not bundled because they are generated from the Flickr8k training data and are too large to reproduce reliably in a small project archive. Run the one-command training workflow after placing the dataset.
