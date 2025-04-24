# LBCE-BERT MVP

A Minimal Viable Project (MVP) based on the LBCE-BERT paper for predicting linear B-cell epitopes from protein sequences using pre-computed BERT embeddings.

## Overview

This project implements a simplified version of the LBCE-BERT model described in the paper [Prediction of linear B-cell epitopes based on protein sequence features and BERT embeddings](https://www.nature.com/articles/s41598-024-53028-w) by Liu et al. The original implementation can be found at [https://github.com/Lfang111/LBCE-BERT](https://github.com/Lfang111/LBCE-BERT).

The MVP focuses on the core functionality:
- Loading pre-computed BERT embeddings
- Implementing feature extraction (AAC, AAP, AAT)
- Creating a prediction pipeline using XGBoost
- Training and evaluating the model on the BCPreds dataset
- Providing a clean prediction interface for new protein sequences

## Project Structure

```
lbce-bert-mvp/
├── data/                      # Data directory
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data files
│   └── embeddings/            # Pre-computed BERT embeddings
├── src/                       # Source code
│   ├── features/              # Feature extraction modules
│   │   ├── __init__.py
│   │   ├── aac.py             # Amino Acid Composition
│   │   ├── aap.py             # Amino Acid Pair antigenicity scale
│   │   ├── aat.py             # Amino Acid Trimer antigenicity scale
│   │   └── bert.py            # BERT embeddings loader
│   ├── models/                # Model implementation
│   │   ├── __init__.py
│   │   └── xgboost_model.py   # XGBoost model implementation
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Data loading utilities
│   │   └── evaluation.py      # Evaluation metrics
│   └── __init__.py
├── notebooks/                 # Jupyter notebooks for demonstration
│   └── demo.ipynb             # Demo notebook
├── scripts/                   # Executable scripts
│   ├── train.py               # Script to train the model
│   ├── predict.py             # Script to make predictions
│   └── evaluate.py            # Script to evaluate model performance
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_utils.py
├── requirements.txt           # Project dependencies
├── setup.py                   # Package setup script
└── README.md                  # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lbce-bert-mvp.git
cd lbce-bert-mvp

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python scripts/train.py --dataset bcpreds --output_model models/lbce_bert_model.pkl
```

### Making Predictions

```bash
python scripts/predict.py --input sequences.txt --model models/lbce_bert_model.pkl --output predictions.csv
```

### Evaluating Model Performance

```bash
python scripts/evaluate.py --test_data data/processed/bcpreds_test.csv --model models/lbce_bert_model.pkl
```

## Data

This project uses the BCPreds dataset, which contains:
- 701 epitope sequences (positive samples)
- 701 non-epitope sequences (negative samples)
- Each sequence is 20 amino acid residues in length

The pre-computed BERT embeddings are loaded from the original repository.

## Features

The model uses the following features:
1. **Amino Acid Composition (AAC)**: Frequency of each amino acid in the sequence
2. **Amino Acid Pair antigenicity scale (AAP)**: Antigenicity values for pairs of amino acids
3. **Amino Acid Trimer antigenicity scale (AAT)**: Antigenicity values for triplets of amino acids
4. **BERT embeddings**: Pre-computed embeddings from a BERT model trained on protein sequences

## Model

The model uses XGBoost, a gradient boosting framework, to predict whether a protein sequence is a linear B-cell epitope. The hyperparameters are simplified based on the values reported in the original paper.

## Evaluation Metrics

The model performance is evaluated using:
- Accuracy
- Precision
- Sensitivity (Recall)
- F1 score
- Matthews Correlation Coefficient (MCC)
- Area Under the Receiver Operating Characteristic curve (AUROC)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Original LBCE-BERT paper: Liu et al., "Prediction of linear B-cell epitopes based on protein sequence features and BERT embeddings", Scientific Reports, 2024
- Original implementation: [https://github.com/Lfang111/LBCE-BERT](https://github.com/Lfang111/LBCE-BERT)
