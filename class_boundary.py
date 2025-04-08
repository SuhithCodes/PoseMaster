import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
train_data = pd.read_csv(r'logs\train_data_angles.csv')
validation_data = pd.read_csv(r'logs\validation_data_angles.csv')
testing_data = pd.read_csv(r'logs\test_data_angles.csv')

# Selecting two features for visualization
selected_features = ['left_elbow_angle', 'right_elbow_angle']

# Separate features and target
X_train = train_data[selected_features]
y_train = train_data['pose']
X_validation = validation_data[selected_features]
y_validation = validation_data['pose']
X_test = testing_data[selected_features]
y_test = testing_data['pose']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# Create a SVM Classifier
clf = svm.SVC(kernel='linear')

# Train the Classifier
clf.fit(X_train_scaled, y_train)

# Evaluate the model on validation set
validation_accuracy = clf.score(X_validation_scaled, y_validation)
print("Validation Accuracy:", validation_accuracy)

# Plot the decision boundary
# Create a mesh to plot in
h = 0.02  # step size in the mesh
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot decision boundary
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=plt.cm.coolwarm)
plt.xlabel('Left Elbow Angle (scaled)')
plt.ylabel('Right Elbow Angle (scaled)')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('SVM Decision Boundary')
plt.show()
