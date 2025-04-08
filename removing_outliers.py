import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv("logs/validation_data.csv")

# Select only numerical columns
numerical_cols = df.select_dtypes(include=[np.number])

# Calculate mean and standard deviation for each numerical column
mean = numerical_cols.mean()
std_dev = numerical_cols.std()

# Define the z-score threshold
threshold = 3

# Calculate z-score for each data point
z_scores = (numerical_cols - mean) / std_dev

# Find rows where any z-score exceeds the threshold
outliers = (z_scores > threshold).any(axis=1)

# Remove outliers from the DataFrame
cleaned_df = df[~outliers]

# Save the cleaned DataFrame to a new CSV file
cleaned_df.to_csv("logs/validation_data_cleaned.csv", index=False)
