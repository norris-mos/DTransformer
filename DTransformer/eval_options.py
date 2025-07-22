import numpy as np
import torch
from sklearn import metrics


class OptionsEvaluator:
    """Evaluator for options tracing (multiclass classification)"""
    
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def evaluate(self, y_true, y_pred):
        """
        Evaluate multiclass predictions for options tracing
        
        Args:
            y_true: ground truth options (1-4 for A,B,C,D)
            y_pred: predicted option probabilities or logits
        """
        mask = y_true >= 0
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Convert predictions to class predictions (0-3)
        if len(y_pred_filtered.shape) > 1 and y_pred_filtered.shape[-1] == 4:
            # If y_pred is probabilities/logits for 4 classes, take argmax
            y_pred_class = torch.argmax(y_pred_filtered, dim=-1) + 1  # Convert back to 1-4
        else:
            # If y_pred is single values, round them
            y_pred_class = y_pred_filtered.round()
        
        self.y_true.extend(y_true_filtered.cpu().tolist())
        self.y_pred.extend(y_pred_class.cpu().tolist())

    def report(self):
        y_true_arr = np.asarray(self.y_true)
        y_pred_arr = np.asarray(self.y_pred)
        
        # Convert to 0-based indexing for sklearn
        y_true_0based = y_true_arr - 1
        y_pred_0based = y_pred_arr - 1
        
        # Ensure valid range (0-3)
        y_true_0based = np.clip(y_true_0based, 0, 3)
        y_pred_0based = np.clip(y_pred_0based, 0, 3)
        
        # Calculate multiclass metrics
        accuracy = metrics.accuracy_score(y_true_0based, y_pred_0based)
        
        # Multiclass precision, recall, F1 (macro averaged)
        precision = metrics.precision_score(y_true_0based, y_pred_0based, 
                                          average='macro', zero_division=0)
        recall = metrics.recall_score(y_true_0based, y_pred_0based, 
                                    average='macro', zero_division=0)
        f1 = metrics.f1_score(y_true_0based, y_pred_0based, 
                             average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = metrics.precision_score(y_true_0based, y_pred_0based, 
                                                     average=None, zero_division=0)
        recall_per_class = metrics.recall_score(y_true_0based, y_pred_0based, 
                                               average=None, zero_division=0)
        f1_per_class = metrics.f1_score(y_true_0based, y_pred_0based, 
                                       average=None, zero_division=0)
        
        # Confusion matrix
        cm = metrics.confusion_matrix(y_true_0based, y_pred_0based, labels=[0, 1, 2, 3])
        
        # For multiclass AUC, use macro-averaged one-vs-rest AUC
        try:
            # Convert to one-hot for multiclass AUC calculation
            y_true_onehot = np.eye(4)[y_true_0based]
            y_pred_onehot = np.eye(4)[y_pred_0based]
            auc_macro = metrics.roc_auc_score(y_true_onehot, y_pred_onehot, 
                                            multi_class='ovr', average='macro')
        except:
            # Fallback if AUC calculation fails
            auc_macro = 0.0

        return {
            "acc": accuracy,
            "auc": auc_macro,  # Add AUC for compatibility
            "precision": precision,  # Add for compatibility
            "recall": recall,        # Add for compatibility
            "f1": f1,               # Add for compatibility
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "precision_per_class": {
                "A": precision_per_class[0] if len(precision_per_class) > 0 else 0,
                "B": precision_per_class[1] if len(precision_per_class) > 1 else 0,
                "C": precision_per_class[2] if len(precision_per_class) > 2 else 0,
                "D": precision_per_class[3] if len(precision_per_class) > 3 else 0,
            },
            "recall_per_class": {
                "A": recall_per_class[0] if len(recall_per_class) > 0 else 0,
                "B": recall_per_class[1] if len(recall_per_class) > 1 else 0,
                "C": recall_per_class[2] if len(recall_per_class) > 2 else 0,
                "D": recall_per_class[3] if len(recall_per_class) > 3 else 0,
            },
            "f1_per_class": {
                "A": f1_per_class[0] if len(f1_per_class) > 0 else 0,
                "B": f1_per_class[1] if len(f1_per_class) > 1 else 0,
                "C": f1_per_class[2] if len(f1_per_class) > 2 else 0,
                "D": f1_per_class[3] if len(f1_per_class) > 3 else 0,
            },
            "confusion_matrix": cm.tolist(),
            # Option distribution
            "true_distribution": np.bincount(y_true_0based, minlength=4) / len(y_true_0based),
            "pred_distribution": np.bincount(y_pred_0based, minlength=4) / len(y_pred_0based),
        }

    def reset(self):
        """Reset the evaluator for a new evaluation"""
        self.y_true = []
        self.y_pred = []