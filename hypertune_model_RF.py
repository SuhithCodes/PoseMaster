import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv(r'logs\train_data_angles.csv')
test_data = pd.read_csv(r'logs\validation_data_angles.csv')

# Separate features and target
X_train, y_train = train_data.drop('pose', axis=1), train_data['pose']
X_test, y_test = test_data.drop('pose', axis=1), test_data['pose']

# Initialize lists to store training and test accuracies
train_accuracies = []
test_accuracies = []

# Define range of n_estimators
n_estimators_range = range(1, 201)

# Train Random Forest models with different n_estimators values
for n_estimators in n_estimators_range:
    # Create Random Forest classifier with current n_estimators
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    
    # Fit the model on training data
    rf_model.fit(X_train, y_train)
    
    # Predict on training data
    y_train_pred = rf_model.predict(X_train)
    
    # Predict on test data
    y_test_pred = rf_model.predict(X_test)
    
    # Calculate training and test accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Append accuracies to the lists
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot training and test accuracies versus number of estimators
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_accuracies, label='Training Accuracy')
plt.plot(n_estimators_range, test_accuracies, label='Test Accuracy')
plt.title('Training and Test Accuracy vs Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
