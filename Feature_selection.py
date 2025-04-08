import numpy as np

def fisher_score(X, y):
    """
    Calculate Fisher score for each feature.
    
    Parameters:
        X (numpy array): Input features.
        y (numpy array): Target labels.
    
    Returns:
        scores (numpy array): Fisher scores for each feature.
    """
    class_labels = np.unique(y)
    scores = []
    for feature in range(X.shape[1]):
        feature_scores = []
        for label in class_labels:
            # Select samples belonging to the current class
            class_samples = X[y == label]
            # Calculate mean and variance for the current feature and class
            feature_mean = np.mean(class_samples[:, feature])
            feature_variance = np.var(class_samples[:, feature])
            feature_scores.append((label, feature_variance))
        # Calculate Fisher score for the current feature
        fisher_score = sum([var for _, var in feature_scores]) / sum([var for _, var in feature_scores])
        scores.append(fisher_score)
    return np.array(scores)

# Importing necessary libraries
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

def fisher_score(X, y):
    """
    Calculate Fisher score for each feature.
    
    Parameters:
        X (numpy array): Input features.
        y (numpy array): Target labels.
    
    Returns:
        scores (numpy array): Fisher scores for each feature.
    """
    class_labels = np.unique(y)
    scores = []
    for feature in range(X.shape[1]):
        feature_scores = []
        for label in class_labels:
            # Select samples belonging to the current class
            class_samples = X[y == label]
            # Calculate mean and variance for the current feature and class
            feature_mean = np.mean(class_samples[:, feature])
            feature_variance = np.var(class_samples[:, feature])
            feature_scores.append((label, feature_variance))
        # Calculate Fisher score for the current feature
        fisher_score = sum([var for _, var in feature_scores]) / sum([var for _, var in feature_scores])
        scores.append(fisher_score)
    return np.array(scores)

# Function to train SVM model and calculate accuracies
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    # Calculate Fisher scores
    fisher_scores = fisher_score(X_train, y_train)

    # Sort features based on Fisher scores
    sorted_indices = np.argsort(fisher_scores)[::-1]

    # Initialize the SVM model
    svm_model = SVC(kernel='rbf', C=10, gamma=0.01, random_state=0)

    # Lists to store accuracies
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    # Range of k values (number of selected features)
    k_values = range(1, len(fisher_scores) )

    # Loop through different values of k
    for k in k_values:
        # Select top k features
        selected_features = sorted_indices[:k]

        # Select only the selected features for training
        X_train_selected = X_train[:, selected_features]

        # Standardize features for training
        scaler = StandardScaler()
        X_train_selected_scaled = scaler.fit_transform(X_train_selected)

        # Train the SVM model
        svm_model.fit(X_train_selected_scaled, y_train)

        # Select only the selected features for validation and testing
        X_val_selected = X_val[:, selected_features]
        X_test_selected = X_test[:, selected_features]

        # Standardize features for validation and testing
        X_val_selected_scaled = scaler.transform(X_val_selected)
        X_test_selected_scaled = scaler.transform(X_test_selected)

        # Predictions for training set
        y_train_pred = svm_model.predict(X_train_selected_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)

        # Predictions for validation set
        y_val_pred = svm_model.predict(X_val_selected_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_accuracies.append(val_accuracy)

        # Predictions for testing set
        y_test_pred = svm_model.predict(X_test_selected_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_accuracy)

    return k_values, train_accuracies, val_accuracies, test_accuracies

# Load the data
train_data_length_ratios = pd.read_csv(r'logs\train_data_length_ratios.csv')
validation_data_length_ratios = pd.read_csv(r'logs\validation_data_length_ratios.csv')
test_data_length_ratios = pd.read_csv(r'logs\test_data_length_ratios.csv')

train_data_angles = pd.read_csv(r'logs\train_data_angles.csv')
validation_data_angles = pd.read_csv(r'logs\validation_data_angles.csv')
test_data_angles = pd.read_csv(r'logs\test_data_angles.csv')

# Separate features and target
X_train_length_ratios, y_train_length_ratios = train_data_length_ratios.drop('pose', axis=1).to_numpy(), train_data_length_ratios['pose'].to_numpy()
X_val_length_ratios, y_val_length_ratios = validation_data_length_ratios.drop('pose', axis=1).to_numpy(), validation_data_length_ratios['pose'].to_numpy()
X_test_length_ratios, y_test_length_ratios = test_data_length_ratios.drop('pose', axis=1).to_numpy(), test_data_length_ratios['pose'].to_numpy()

X_train_angles, y_train_angles = train_data_angles.drop('pose', axis=1).to_numpy(), train_data_angles['pose'].to_numpy()
X_val_angles, y_val_angles = validation_data_angles.drop('pose', axis=1).to_numpy(), validation_data_angles['pose'].to_numpy()
X_test_angles, y_test_angles = test_data_angles.drop('pose', axis=1).to_numpy(), test_data_angles['pose'].to_numpy()

# Train and evaluate SVM models for both datasets
k_values_length_ratios, train_accuracies_length_ratios, val_accuracies_length_ratios, test_accuracies_length_ratios = train_and_evaluate(X_train_length_ratios, y_train_length_ratios, X_val_length_ratios, y_val_length_ratios, X_test_length_ratios, y_test_length_ratios)

k_values_angles, train_accuracies_angles, val_accuracies_angles, test_accuracies_angles = train_and_evaluate(X_train_angles, y_train_angles, X_val_angles, y_val_angles, X_test_angles, y_test_angles)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot for length ratios
axs[0, 0].plot(k_values_length_ratios, train_accuracies_length_ratios, label='Training Accuracy')
axs[0, 0].plot(k_values_length_ratios, val_accuracies_length_ratios, label='Validation Accuracy')
axs[0, 0].plot(k_values_length_ratios, test_accuracies_length_ratios, label='Testing Accuracy')
axs[0, 0].set_title('Length Ratios Dataset')
axs[0, 0].set_xlabel('Number of Features')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot for angles
axs[0, 1].plot(k_values_angles, train_accuracies_angles, label='Training Accuracy')
axs[0, 1].plot(k_values_angles, val_accuracies_angles, label='Validation Accuracy')
axs[0, 1].plot(k_values_angles, test_accuracies_angles, label='Testing Accuracy')
axs[0, 1].set_title('Angles Dataset')
axs[0, 1].set_xlabel('Number of Features')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Hide the empty subplot
axs[1, 0].axis('off')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()
