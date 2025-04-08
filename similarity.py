import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('logs/train_data_angles.csv')

# Extract poses and their corresponding angles
poses = data['pose']
angles = data.drop('pose', axis=1)

# Calculate the mean angles for each pose
mean_angles = angles.groupby(poses).mean()

# Discretize the mean angles into bins
num_bins = 10  # You can adjust the number of bins as needed
discretized_mean_angles = pd.DataFrame()

# Discretize each mean angle individually
for col in mean_angles.columns:
    discretized_angle = pd.cut(mean_angles[col], bins=num_bins, labels=False)
    discretized_mean_angles[col] = discretized_angle

# Calculate Jaccard similarity for each pair of poses
similarity_matrix = np.zeros((len(mean_angles), len(mean_angles)))
for i in range(len(mean_angles)):
    for j in range(len(mean_angles)):
        if i == j:
            similarity_matrix[i, j] = 1  # Set diagonal elements to 1 for identical poses (same class)
        else:
            pose1 = discretized_mean_angles.iloc[i]
            pose2 = discretized_mean_angles.iloc[j]
            intersection = np.sum((pose1 == pose2) & (pose1 >= 0) & (pose2 >= 0))
            union = np.sum((pose1 >= 0) | (pose2 >= 0))
            similarity = intersection / union
            similarity_matrix[i, j] = similarity

# Plotting the similarity matrix
plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Jaccard Similarity')
plt.xticks(np.arange(len(mean_angles)), mean_angles.index, rotation=90)
plt.yticks(np.arange(len(mean_angles)), mean_angles.index)
plt.title('Jaccard Similarity Matrix')
plt.xlabel('Poses')
plt.ylabel('Poses')
plt.show()
