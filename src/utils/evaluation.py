"""
Evaluation metrics for LBCE-BERT MVP.

This module provides functions for evaluating the performance of the LBCE-BERT model.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve, confusion_matrix
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate evaluation metrics for binary classification.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray, optional
        Predicted probabilities for the positive class.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),  # Sensitivity is the same as recall
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }
    
    # Calculate AUROC if probabilities are provided
    if y_prob is not None:
        metrics['auroc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary containing evaluation metrics.
    """
    print("Evaluation Metrics:")
    print(f"  Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision:   {metrics.get('precision', 0):.4f}")
    print(f"  Sensitivity: {metrics.get('sensitivity', 0):.4f}")
    print(f"  F1 Score:    {metrics.get('f1', 0):.4f}")
    print(f"  MCC:         {metrics.get('mcc', 0):.4f}")
    if 'auroc' in metrics:
        print(f"  AUROC:       {metrics.get('auroc', 0):.4f}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_file: Optional[str] = None) -> None:
    """
    Plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    output_file : str, optional
        Path to save the plot. If None, the plot is displayed but not saved.
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save or display plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_file: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    output_file : str, optional
        Path to save the plot. If None, the plot is displayed but not saved.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], ['Negative', 'Positive'])
    plt.yticks([0.5, 1.5], ['Negative', 'Positive'])
    
    # Save or display plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_metrics_to_file(metrics: Dict[str, float], output_file: str) -> None:
    """
    Save evaluation metrics to a text file.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary containing evaluation metrics.
    output_file : str
        Path to save the metrics.
    """
    with open(output_file, 'w') as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"  Accuracy:    {metrics.get('accuracy', 0):.4f}\n")
        f.write(f"  Precision:   {metrics.get('precision', 0):.4f}\n")
        f.write(f"  Sensitivity: {metrics.get('sensitivity', 0):.4f}\n")
        f.write(f"  F1 Score:    {metrics.get('f1', 0):.4f}\n")
        f.write(f"  MCC:         {metrics.get('mcc', 0):.4f}\n")
        if 'auroc' in metrics:
            f.write(f"  AUROC:       {metrics.get('auroc', 0):.4f}\n")
