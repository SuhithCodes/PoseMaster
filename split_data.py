import os
import shutil
import random

# Define the paths
source_folder = "data/Yoga-82/yoga_dataset_images"
train_folder = "data/TRAIN"
val_folder = "data/VAL"
test_folder = "data/TEST"

# Create train, validation, and test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Iterate through each yoga position folder
for yoga_position_folder in os.listdir(source_folder):
    # Create train, validation, and test folders for each yoga position
    os.makedirs(os.path.join(train_folder, yoga_position_folder), exist_ok=True)
    os.makedirs(os.path.join(val_folder, yoga_position_folder), exist_ok=True)
    os.makedirs(os.path.join(test_folder, yoga_position_folder), exist_ok=True)
    
    # Get list of images in the current yoga position folder
    images = os.listdir(os.path.join(source_folder, yoga_position_folder))
    
    # Shuffle the list of images to randomize
    random.shuffle(images)
    
    # Calculate the number of images for each set
    num_total_images = len(images)
    num_train_images = int(num_total_images * 0.7)
    num_val_images = int(num_total_images * 0.2)
    num_test_images = num_total_images - num_train_images - num_val_images
    
    # Move images to the train set
    for image in images[:num_train_images]:
        source_path = os.path.join(source_folder, yoga_position_folder, image)
        dest_path = os.path.join(train_folder, yoga_position_folder, image)
        shutil.move(source_path, dest_path)
    
    # Move images to the validation set
    for image in images[num_train_images:num_train_images + num_val_images]:
        source_path = os.path.join(source_folder, yoga_position_folder, image)
        dest_path = os.path.join(val_folder, yoga_position_folder, image)
        shutil.move(source_path, dest_path)
    
    # Move images to the test set
    for image in images[num_train_images + num_val_images:]:
        source_path = os.path.join(source_folder, yoga_position_folder, image)
        dest_path = os.path.join(test_folder, yoga_position_folder, image)
        shutil.move(source_path, dest_path)

print("Data split into training, validation, and testing sets successfully.")
