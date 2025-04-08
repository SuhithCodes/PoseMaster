import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv(r'logs\train_data_angles.csv')
test_data = pd.read_csv(r'logs\validation_data_angles.csv')

# Separate features and target
X_train, y_train = train_data.drop('pose', axis=1), train_data['pose']
X_test, y_test = test_data.drop('pose', axis=1), test_data['pose']

# Train SVC with default parameters
svc_default = SVC()
svc_default.fit(X_train, y_train)
y_pred_default = svc_default.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# Train SVC with customized parameters
svc_custom = SVC(kernel='poly', C=18, degree=2)
svc_custom.fit(X_train, y_train)
y_pred_custom = svc_custom.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)

# Plot before and after accuracy
labels = ['Before (Default Parameters)', 'After (Custom Parameters)']
accuracies = [accuracy_default, accuracy_custom]

plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=accuracies)
plt.ylabel('Accuracy')
plt.title('Before and After Accuracy with SVC')

# Display accuracy values on top of the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center')

plt.ylim(0, 1)
plt.show()
