# Simple LBCE-BERT

A simplified implementation of the LBCE-BERT model for predicting linear B-cell epitopes using pre-computed BERT embeddings.

## Overview

This project implements a simplified version of the LBCE-BERT model described in the paper [Prediction of linear B-cell epitopes based on protein sequence features and BERT embeddings](https://www.nature.com/articles/s41598-024-53028-w) by Liu et al. The original implementation can be found at [https://github.com/Lfang111/LBCE-BERT](https://github.com/Lfang111/LBCE-BERT).

This simplified version focuses on:
- Using only the pre-computed BERT embeddings from the original repository
- Training an XGBoost model on these embeddings
- Providing scripts for training, prediction, and evaluation

## Project Structure

```
simple-lbce-bert/
├── data/                      # Data directory
│   └── CLS_fea.txt            # Pre-computed BERT embeddings
├── src/                       # Source code
│   ├── data_utils.py          # Data loading and processing utilities
│   ├── model.py               # XGBoost model implementation
│   └── evaluation.py          # Evaluation metrics
├── notebooks/                 # Jupyter notebooks
│   └── demo.ipynb             # Demo notebook
├── scripts/                   # Executable scripts
│   ├── train.py               # Script to train the model
│   ├── predict.py             # Script to make predictions
│   └── evaluate.py            # Script to evaluate model performance
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/simple-lbce-bert.git
cd simple-lbce-bert

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python scripts/train.py --data_file data/CLS_fea.txt --output_model models/lbce_bert_model.pkl
```

### Making Predictions

```bash
python scripts/predict.py --data_file data/test_data.txt --model models/lbce_bert_model.pkl --output predictions.csv
```

### Evaluating Model Performance

```bash
python scripts/evaluate.py --data_file data/test_data.txt --model models/lbce_bert_model.pkl
```

## Data

This project uses the pre-computed BERT embeddings from the BCPreds dataset in the original LBCE-BERT repository. The embeddings are stored in the `CLS_fea.txt` file.

## Model

The model uses XGBoost, a gradient boosting framework, to predict whether a protein sequence is a linear B-cell epitope based on the pre-computed BERT embeddings.

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
