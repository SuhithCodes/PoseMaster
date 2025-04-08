import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe modules
mp_pose = mp.solutions.pose

def rename_columns(df):
    column_mapping = {
        'folder_name': 'pose',
        'landmark_11_x': 'left_shoulder_x', 'landmark_11_y': 'left_shoulder_y',
        'landmark_12_x': 'right_shoulder_x', 'landmark_12_y': 'right_shoulder_y',
        'landmark_13_x': 'left_elbow_x', 'landmark_13_y': 'left_elbow_y',
        'landmark_14_x': 'right_elbow_x', 'landmark_14_y': 'right_elbow_y',
        'landmark_15_x': 'left_wrist_x', 'landmark_15_y': 'left_wrist_y',
        'landmark_16_x': 'right_wrist_x', 'landmark_16_y': 'right_wrist_y',
        'landmark_23_x': 'left_hip_x', 'landmark_23_y': 'left_hip_y',
        'landmark_24_x': 'right_hip_x', 'landmark_24_y': 'right_hip_y',
        'landmark_25_x': 'left_knee_x', 'landmark_25_y': 'left_knee_y',
        'landmark_26_x': 'right_knee_x', 'landmark_26_y': 'right_knee_y',
        'landmark_27_x': 'left_ankle_x', 'landmark_27_y': 'left_ankle_y',
        'landmark_28_x': 'right_ankle_x', 'landmark_28_y': 'right_ankle_y'
    }

    df.rename(columns=column_mapping, inplace=True)

def extract_landmark_values(results):
    landmark_values = {}
    for landmark in [11, 12, 13, 14, 15, 16,23, 24, 25, 26, 27, 28]:
        landmark_values[f"landmark_{landmark}_x"] = results.pose_landmarks.landmark[landmark].x
        landmark_values[f"landmark_{landmark}_y"] = results.pose_landmarks.landmark[landmark].y
    return landmark_values

def process_image(image_path, pose):
    try:
        print("Processing image:", image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)  # Assignment of 'results' here
        print("Image processed successfully")
        return results
    except Exception as e:
        print("Error processing image:", e)
        return None  # Return None if an error occurs


def main(base_dataset_path, output_csv_path):
    # Initialize the pose estimator
    with mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as pose:

        all_data = []

        # Loop over each folder in the dataset directory
        for folder_name in os.listdir(base_dataset_path):
            folder_path = os.path.join(base_dataset_path, folder_name)
            if os.path.isdir(folder_path):
                print("Processing folder:", folder_path)
                # Loop over each image in the current folder
                for filename in os.listdir(folder_path):
                    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                        image_path = os.path.join(folder_path, filename)
                        results = process_image(image_path, pose)
                        if results is not None and results.pose_landmarks is not None:
                            print("Landmarks detected in the image:", filename)
                            landmark_values = extract_landmark_values(results)
                            landmark_values["folder_name"] = folder_name
                            all_data.append(landmark_values)
                        else:
                            print("No landmarks detected in the image:", filename)

        df = pd.DataFrame(all_data)
        df = df[['folder_name'] + [col for col in df.columns if col != 'folder_name']]
        rename_columns(df)
        df.to_csv(output_csv_path, index=False)
        print("Landmark values saved to:", output_csv_path)

if __name__ == "__main__":
    # Path to the directory containing training data
    train_dataset_path = r"data\TRAIN"
    train_output_csv_path = r"logs\train_data.csv"
    # Rename columns
    main(train_dataset_path, train_output_csv_path)

    # Path to the directory containing testing data
    test_dataset_path = r"data\TEST"
    test_output_csv_path = r"logs\test_data.csv"
    main(test_dataset_path, test_output_csv_path)

    # Path to the directory containing validation data
    test_dataset_path = r"data\VALIDATION"
    test_output_csv_path = r"logs\validation_data.csv"
    main(test_dataset_path, test_output_csv_path)
