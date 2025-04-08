import pandas as pd
import math

def compute_length_ratios(data):
    features = {}
    
    # Apply the function to each row using a lambda function
    data['forearm_upper_arm_ratio_left'] = data.apply(lambda row: calculate_length_ratio(row['left_elbow_x'], row['left_elbow_y'],
                                                                                        row['left_wrist_x'], row['left_wrist_y'],
                                                                                        row['left_shoulder_x'], row['left_shoulder_y']), axis=1)

    data['forearm_upper_arm_ratio_right'] = data.apply(lambda row: calculate_length_ratio(row['right_elbow_x'], row['right_elbow_y'],
                                                                                        row['right_wrist_x'], row['right_wrist_y'],
                                                                                        row['right_shoulder_x'], row['right_shoulder_y']), axis=1)

    data['thigh_lower_leg_ratio_left'] = data.apply(lambda row: calculate_length_ratio(row['left_knee_x'], row['left_knee_y'],
                                                                                        row['left_ankle_x'], row['left_ankle_y'],
                                                                                        row['left_hip_x'], row['left_hip_y']), axis=1)

    data['thigh_lower_leg_ratio_right'] = data.apply(lambda row: calculate_length_ratio(row['right_knee_x'], row['right_knee_y'],
                                                                                        row['right_ankle_x'], row['right_ankle_y'],
                                                                                        row['right_hip_x'], row['right_hip_y']), axis=1)
    
    data['upper_arm_upper_leg_ratio_left'] = data.apply(lambda row: calculate_length_ratio(row['left_shoulder_x'], row['left_shoulder_y'],
                                                                                        row['left_elbow_x'], row['left_elbow_y'],
                                                                                        row['left_hip_x'], row['left_hip_y']), axis=1)

    data['upper_arm_upper_leg_ratio_right'] = data.apply(lambda row: calculate_length_ratio(row['right_shoulder_x'], row['right_shoulder_y'],
                                                                                        row['right_elbow_x'], row['right_elbow_y'],
                                                                                        row['right_hip_x'], row['right_hip_y']), axis=1)

    data['forearm_lower_leg_ratio_left'] = data.apply(lambda row: calculate_length_ratio(row['left_elbow_x'], row['left_elbow_y'],
                                                                                        row['left_wrist_x'], row['left_wrist_y'],
                                                                                        row['left_knee_x'], row['left_knee_y']), axis=1)

    data['forearm_lower_leg_ratio_right'] = data.apply(lambda row: calculate_length_ratio(row['right_elbow_x'], row['right_elbow_y'],
                                                                                        row['right_wrist_x'], row['right_wrist_y'],
                                                                                        row['right_knee_x'], row['right_knee_y']), axis=1)

    data['left_right_upper_arm_ratio'] = data.apply(lambda row: calculate_length_ratio(row['left_shoulder_x'], row['left_shoulder_y'],
                                                                                        row['left_elbow_x'], row['left_elbow_y'],
                                                                                        row['right_shoulder_x'], row['right_shoulder_y']), axis=1)

    data['left_right_upper_leg_ratio'] = data.apply(lambda row: calculate_length_ratio(row['left_hip_x'], row['left_hip_y'],
                                                                                        row['left_knee_x'], row['left_knee_y'],
                                                                                        row['right_hip_x'], row['right_hip_y']), axis=1)

    data['left_right_upper_body_ratio'] = data.apply(lambda row: calculate_length_ratio(row['left_shoulder_x'], row['left_shoulder_y'],
                                                                                      row['left_hip_x'], row['left_hip_y'],
                                                                                      row['right_shoulder_x'], row['right_shoulder_y']), axis=1)

    return data[['pose', 'forearm_upper_arm_ratio_left', 'forearm_upper_arm_ratio_right', 'thigh_lower_leg_ratio_left', 'thigh_lower_leg_ratio_right', 'upper_arm_upper_leg_ratio_left', 'upper_arm_upper_leg_ratio_right', 'forearm_lower_leg_ratio_left', 'forearm_lower_leg_ratio_right','left_right_upper_arm_ratio','left_right_upper_leg_ratio','left_right_upper_body_ratio']]


def calculate_length_ratio(x1, y1, x2, y2, x3, y3):
    # Calculate ratio of lengths of two line segments
    length1 = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    length2 = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)
    if length2 == 0:  # Avoid division by zero
        return float('nan')
    ratio = length1 / length2
    return ratio


def calculate_angle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
    '''
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle

def calculate_landmark_angles(data):
    '''
    Calculate angles between specified landmarks for each row in the DataFrame.
    Args:
        data: DataFrame containing landmark coordinates.
    Returns:
        DataFrame: New DataFrame containing pose column and calculated angles.
    '''
    data['left_elbow_angle'] = data.apply(lambda row: calculate_angle((row['left_shoulder_x'], row['left_shoulder_y'], 0),
                                                                      (row['left_elbow_x'], row['left_elbow_y'], 0),
                                                                      (row['left_wrist_x'], row['left_wrist_y'], 0)), axis=1)

    data['right_elbow_angle'] = data.apply(lambda row: calculate_angle((row['right_shoulder_x'], row['right_shoulder_y'], 0),
                                                                       (row['right_elbow_x'], row['right_elbow_y'], 0),
                                                                       (row['right_wrist_x'], row['right_wrist_y'], 0)), axis=1)

    data['left_shoulder_angle'] = data.apply(lambda row: calculate_angle((row['left_hip_x'], row['left_hip_y'], 0),
                                                                         (row['left_shoulder_x'], row['left_shoulder_y'], 0),
                                                                         (row['left_elbow_x'], row['left_elbow_y'], 0)), axis=1)

    data['right_shoulder_angle'] = data.apply(lambda row: calculate_angle((row['right_hip_x'], row['right_hip_y'], 0),
                                                                          (row['right_shoulder_x'], row['right_shoulder_y'], 0),
                                                                          (row['right_elbow_x'], row['right_elbow_y'], 0)), axis=1)

    data['left_hip_angle'] = data.apply(lambda row: calculate_angle((row['left_shoulder_x'], row['left_shoulder_y'], 0),
                                                                         (row['left_hip_x'], row['left_hip_y'], 0),
                                                                         (row['left_knee_x'], row['left_knee_y'], 0)), axis=1)

    data['right_hip_angle'] = data.apply(lambda row: calculate_angle((row['right_shoulder_x'], row['right_shoulder_y'], 0),
                                                                          (row['right_hip_x'], row['right_hip_y'], 0),
                                                                          (row['right_knee_x'], row['right_knee_y'], 0)), axis=1)

    data['left_knee_angle'] = data.apply(lambda row: calculate_angle((row['left_hip_x'], row['left_hip_y'], 0),
                                                                     (row['left_knee_x'], row['left_knee_y'], 0),
                                                                     (row['left_ankle_x'], row['left_ankle_y'], 0)), axis=1)

    data['right_knee_angle'] = data.apply(lambda row: calculate_angle((row['right_hip_x'], row['right_hip_y'], 0),
                                                                      (row['right_knee_x'], row['right_knee_y'], 0),
                                                                      (row['right_ankle_x'], row['right_ankle_y'], 0)), axis=1)

    return data[['pose', 'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 'right_shoulder_angle',  'left_knee_angle', 'right_knee_angle','left_hip_angle','right_hip_angle']]


# Read the CSV file into a DataFrame
data = pd.read_csv('logs/test_data_cleaned.csv')
# Calculate length ratios
length_ratios_data = compute_length_ratios(data)
# Write the new DataFrame to a CSV file
length_ratios_data.to_csv('logs/test_data_length_ratios.csv', index=False)
# Calculate landmark angles
landmark_angles_data = calculate_landmark_angles(data)
# Write the new DataFrame to a CSV file
landmark_angles_data.to_csv('logs/test_data_angles.csv', index=False)
# Merge the DataFrames on the 'pose' column
combined_data = pd.merge(length_ratios_data, landmark_angles_data, on='pose')
# Write the new DataFrame to a CSV file
combined_data.to_csv('logs/test_data_combined.csv', index=False)

# Repeat the process for train_data.csv and validation_data.csv
# Read the CSV file into a DataFrame
data = pd.read_csv('logs/train_data_cleaned.csv')
# Calculate length ratios
length_ratios_data = compute_length_ratios(data)
# Write the new DataFrame to a CSV file
length_ratios_data.to_csv('logs/train_data_length_ratios.csv', index=False)
# Calculate landmark angles
landmark_angles_data = calculate_landmark_angles(data)
# Write the new DataFrame to a CSV file
landmark_angles_data.to_csv('logs/train_data_angles.csv', index=False)
# Merge the DataFrames on the 'pose' column
combined_data = pd.merge(length_ratios_data, landmark_angles_data, on='pose')
# Write the new DataFrame to a CSV file
combined_data.to_csv('logs/train_data_combined.csv', index=False)

# Read the CSV file into a DataFrame
data = pd.read_csv('logs/validation_data_cleaned.csv')
# Calculate length ratios
length_ratios_data = compute_length_ratios(data)
# Write the new DataFrame to a CSV file
length_ratios_data.to_csv('logs/validation_data_length_ratios.csv', index=False)
# Calculate landmark angles
landmark_angles_data = calculate_landmark_angles(data)
# Write the new DataFrame to a CSV file
landmark_angles_data.to_csv('logs/validation_data_angles.csv', index=False)
# Merge the DataFrames on the 'pose' column
combined_data = pd.merge(length_ratios_data, landmark_angles_data, on='pose')
# Write the new DataFrame to a CSV file
combined_data.to_csv('logs/validation_data_combined.csv', index=False)