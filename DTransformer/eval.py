import numpy as np
import torch
from sklearn import metrics


class Evaluator:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def evaluate(self, y_true, y_pred):
        mask = y_true >= 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        self.y_true.extend(y_true.cpu().tolist())
        self.y_pred.extend(y_pred.cpu().tolist())

    def report(self):
        y_true_binary = np.asarray(self.y_true)
        y_pred_binary = np.asarray(self.y_pred).round()
        
        # Calculate precision, recall, and F1 score
        precision = metrics.precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = metrics.recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = metrics.f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        return {
            "acc": metrics.accuracy_score(y_true_binary, y_pred_binary),
            "auc": metrics.roc_auc_score(self.y_true, self.y_pred),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mae": metrics.mean_absolute_error(self.y_true, self.y_pred),
            "rmse": metrics.mean_squared_error(self.y_true, self.y_pred) ** 0.5,
        }
