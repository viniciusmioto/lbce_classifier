# LBCE-BERT MVP

A Minimal Viable Project (MVP) based on the LBCE-BERT paper for predicting linear B-cell epitopes from protein sequences using pre-computed BERT embeddings.

## Overview

This project implements a simplified version of the LBCE-BERT model described in the paper [Prediction of linear B-cell epitopes based on protein sequence features and BERT embeddings](https://www.nature.com/articles/s41598-024-53028-w) by Liu et al. The original implementation is available at [https://github.com/Lfang111/LBCE-BERT](https://github.com/Lfang111/LBCE-BERT).

Key features:
- Uses pre-computed BERT embeddings for protein sequences
- Extracts AAC, AAP, and AAT features
- Trains an XGBoost model to predict epitopes
- Includes scripts for training, prediction, and evaluation

## Project Structure

```
lbce-bert-mvp/
├── data/
│   ├── proteins/              # Input protein sequences (.txt)
│   └── embeddings/            # Pre-computed BERT embeddings (.txt)
├── src/
│   ├── features/              # Feature extraction modules
│   ├── models/                # XGBoost model wrapper
│   └── utils/                 # Data loading and evaluation
├── scripts/
│   ├── train.py               # Model training script
│   ├── predict.py             # Prediction script
│   └── evaluate.py            # Evaluation script
├── requirements.txt           # Python dependencies
├── setup.py                   # Setup file
└── README.md                  # Project documentation
```

## Installation

```bash
git clone https://github.com/viniciusmioto/lbce_classifier.git
cd lbce_classifier

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Training the Model

To train a model on a dataset with pre-computed BERT embeddings and protein sequence files:

```bash
python scripts/train.py \
    --sequences_dir data/proteins \
    --embeddings_dir data/embeddings \
    --labels_file data/processed/labels.csv \
    --output_model models/lbce_bert_model.pkl
```

**Arguments**:
- `--sequences_dir`: Directory with protein sequence `.txt` files.
- `--embeddings_dir`: Directory with corresponding BERT embedding `.txt` files.
- `--labels_file`: CSV file mapping sequence filenames to binary labels (0 = non-epitope, 1 = epitope).
- `--output_model`: Path where the trained model will be saved.

### Making Predictions

```bash
python scripts/predict.py \
    --sequences_dir data/proteins \
    --embeddings_dir data/embeddings \
    --model models/lbce_bert_model.pkl \
    --output predictions.csv
```

### Evaluating Model Performance

```bash
python scripts/evaluate.py \
    --sequences_dir data/proteins \
    --embeddings_dir data/embeddings \
    --labels_file data/processed/test_labels.csv \
    --model models/lbce_bert_model.pkl
```

## Data Format

- Sequence files: Plain text `.txt` files with a single amino acid sequence per file.
- Embedding files: Matching `.txt` files containing BERT embeddings (one per sequence).
- Label file: A CSV with two columns: `filename` and `label`.

Example:

```csv
filename,label
P12345.txt,1
Q67890.txt,0
```

## Features Used

1. **AAC (Amino Acid Composition)**
2. **AAP (Amino Acid Pair Antigenicity)**
3. **AAT (Amino Acid Trimer Antigenicity)**
4. **BERT Embeddings** (precomputed)

## Model

- **Algorithm**: XGBoost
- **Purpose**: Binary classification (epitope vs. non-epitope)
- **Tuning**: Uses default/simplified hyperparameters inspired by the original paper.

## Evaluation Metrics

- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- Matthews Correlation Coefficient (MCC)
- AUROC (Area Under ROC Curve)

## License

MIT License – see [LICENSE](LICENSE) for details.

## Acknowledgements

- Liu et al., “Prediction of linear B-cell epitopes based on protein sequence features and BERT embeddings”, *Scientific Reports*, 2024.
- Original implementation: [https://github.com/Lfang111/LBCE-BERT](https://github.com/Lfang111/LBCE-BERT)

