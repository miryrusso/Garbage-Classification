# Garbage-Classification
# Garbage Classifier

## Overview

This project trains a convolutional neural network to classify waste images into six categories (**glass, paper, cardboard, plastic, metal, trash**). It serves as a proof‑of‑concept for smart‑bin sorting systems and as an educational example of transfer learning with PyTorch.

## Dataset

We use the Kaggle **Garbage Classification** dataset (≈2.4 k images, 224×224 px, six balanced classes). Download it manually or with the helper in `database/database.py`, then arrange it as:

```
data/
├── glass/
├── paper/
├── cardboard/
├── plastic/
├── metal/
└── trash/
```

Alternatively, provide a CSV mapping image paths to labels (see the `CSVImageDataset` in `src/pre_processing.py`).

## Project Layout

```
project-garbage-classifier/
│
├── README.md          ← *you are here*
├── requirements.txt   ← pip dependencies
│
├── database/
│   └── database.py    ← dataset download / CSV utilities
│
├── src/
│   ├── pre_processing.py  ← data transforms, Dataset class, split logic
│   ├── classifier.py      ← model, training & evaluation loops
│   └── main.py            ← command‑line entry point
│
└── .gitignore
```

## Quick Start

```bash
# 1. create & activate virtual env
python3 -m venv .venv && source .venv/bin/activate

# 2. install deps
pip install -r requirements.txt

# 3. train for 10 epochs\python src/main.py --data_dir data --epochs 10 --batch_size 32

# 4. evaluate on a folder of images
python src/main.py --mode eval --checkpoint checkpoints/best.pt --data_dir data/test
```

Run `python src/main.py --help` for the full CLI.

## Model

We fine‑tune **EfficientNet‑B0** (via `timm`) initialised with ImageNet weights. Change `--model_name` to experiment with other backbones.

## Reproducibility

* Seed fixed at `42` (`torch.manual_seed(42)`).
* Split: 65 % train, 15 % val, 20 % test (see `split_train_val_test`).

## Results (reference)

| Metric   | Validation |   Test |
| -------- | ---------: | -----: |
| Accuracy |     ≈ 90 % | ≈ 89 % |

Confusion matrix and per‑class F1 are available via `classifier.plot_confusion_matrix` and `plot_metrics_bar`.

## License

MIT. Dataset licensed by original Kaggle contributors.

## Acknowledgements

Thanks to the Kaggle dataset authors and the `timm` library.
