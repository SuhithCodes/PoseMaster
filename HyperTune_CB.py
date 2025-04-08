from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate some example data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost classifier
model = CatBoostClassifier(iterations=1000, eval_metric='Accuracy', random_state=42)

# Fit the model
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

# Get the best iteration
best_iteration = model.get_best_iteration()

# Evaluate on test set to check accuracy
test_accuracy = model.score(X_test, y_test)

# Check if test accuracy meets the threshold
if test_accuracy >= 0.92:
    print("Training stopped as accuracy reached 92%.")
else:
    print("Training completed with accuracy:", test_accuracy)
