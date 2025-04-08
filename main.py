import pandas as pd
import numpy as np
from Feature_selection import fisher_score
from model import train_stacking_model
from evaluate_model import evaluate_model, plot_confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler

# Load the data
train_data = pd.read_csv(r'logs\train_data_angles.csv')
validation_data = pd.read_csv(r'logs\validation_data_angles.csv')
testing_data = pd.read_csv(r'logs\test_data_angles.csv')

# Separate features and target
X_train, y_train = train_data.drop('pose', axis=1), train_data['pose']

# Convert the data to numpy arrays
X_train_array = X_train.to_numpy()
y_train_array = y_train.to_numpy()

# Calculate Fisher scores
fisher_scores = fisher_score(X_train_array, y_train_array)

# Sort features based on Fisher scores
sorted_indices = np.argsort(fisher_scores)[::-1]

# Select top k features (e.g., top 8 features)
k = 8
selected_features = sorted_indices[:k]
print("Selected features:", selected_features)

# Train models
stacking_model = train_stacking_model(X_train, y_train, selected_features)

# Separate features and target for validation and testing
X_val, y_val = validation_data.drop('pose', axis=1), validation_data['pose']
X_test, y_test = testing_data.drop('pose', axis=1), testing_data['pose']

# Standardize features for validation and testing
scaler = StandardScaler()
scaler.fit(X_train.iloc[:, selected_features])

# Evaluation
classification_rep_train_stacking, conf_mat_train_stacking = evaluate_model(stacking_model, X_train, y_train, scaler, selected_features)
# Evaluation
classification_rep_val_stacking, conf_mat_val_stacking = evaluate_model(stacking_model, X_val, y_val, scaler, selected_features)
# Evaluation
classification_rep_test_stacking, conf_mat_test_stacking = evaluate_model(stacking_model, X_test, y_test, scaler, selected_features)

# Plot confusion matrices
plot_confusion_matrix(conf_mat_train_stacking, stacking_model.classes_, 'Confusion Matrix - Training Data (Stacking)')
# Plot confusion matrices
plot_confusion_matrix(conf_mat_val_stacking, stacking_model.classes_, 'Confusion Matrix - Validation Data (Stacking)')
# Plot confusion matrices
plot_confusion_matrix(conf_mat_test_stacking, stacking_model.classes_, 'Confusion Matrix - Testing Data (Stacking)')

# Print classification reports
print("\nClassification Report - Training Data (Stacking):\n", classification_rep_train_stacking)
# Print classification reports
print("\nClassification Report - Validation Data (Stacking):\n", classification_rep_val_stacking)
# Print classification reports
print("\nClassification Report - Testing Data (Stacking):\n", classification_rep_test_stacking)

# Save the trained models
joblib.dump(stacking_model, 'models/stacking_model.pkl')

