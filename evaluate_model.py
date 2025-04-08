import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def evaluate_model(model, X_val, y_val, scaler, selected_features):
    # Select only the selected features for validation
    X_val_selected = X_val.iloc[:, selected_features]
    
    # Standardize features for validation
    X_val_selected_scaled = scaler.transform(X_val_selected)
    
    # Predictions for validation set
    y_val_pred = model.predict(X_val_selected_scaled)
    
    # Classification report for validation set
    classification_rep_val = classification_report(y_val, y_val_pred)
    
    # Confusion matrix for validation set
    conf_mat_val = confusion_matrix(y_val, y_val_pred)
    
    return classification_rep_val, conf_mat_val

import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(conf_mat, classes, title, ax=None):
    if ax is None:
        plt.figure(figsize=(8, 8))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.show()
    else:
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
