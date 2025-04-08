import cv2
import numpy as np
import os

# Function to perform data augmentation on an image
def augment_image(image, save_dir, filename_prefix):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Flip the image horizontally
    flipped_image = cv2.flip(image, 1)
    save_path = os.path.join(save_dir, filename_prefix + "_flipped.jpg")
    cv2.imwrite(save_path, flipped_image)

    # Scale the image
    scale_factor = np.random.uniform(0.8, 1.2)
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    save_path = os.path.join(save_dir, filename_prefix + "_scaled.jpg")
    cv2.imwrite(save_path, scaled_image)

    # Add random Gaussian noise
    mean = 0
    stddev = 25
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    save_path = os.path.join(save_dir, filename_prefix + "_noisy.jpg")
    cv2.imwrite(save_path, noisy_image)

    # Adjust brightness and contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 30    # Brightness control (0-100)
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    save_path = os.path.join(save_dir, filename_prefix + "_adjusted.jpg")
    cv2.imwrite(save_path, adjusted_image)

# Function to loop through a folder of images and augment each image
def augment_images_in_folder(input_folder, output_folder, max_files):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files_created = 0  # Keep track of the total number of files created
    
    # Loop through each image file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Perform data augmentation
            filename_prefix = os.path.splitext(filename)[0]
            augment_image(image, output_folder, filename_prefix)
            
            files_created += 4  # Each image is augmented into 6 different files
            
            # Stop augmentation if the total files created reach the limit
            if files_created >= max_files:
                break

# Example usage
if __name__ == "__main__":
    # Specify input and output folders
    input_folder = "data\VALIDATION\GODDESS"
    output_folder = "data\VALIDATION\GODDESS"
    max_files = 150
    # Perform data augmentation on images in the input folder
    augment_images_in_folder(input_folder, output_folder, max_files)

    # Specify input and output folders
    input_folder = "data\VALIDATION\PLANK"
    output_folder = "data\VALIDATION\PLANK"
    max_files = 150
    # Perform data augmentation on images in the input folder
    augment_images_in_folder(input_folder, output_folder, max_files)

    # Specify input and output folders
    input_folder = "data\VALIDATION\WARRIOR_2"
    output_folder = "data\VALIDATION\WARRIOR_2"
    max_files = 150
    # Perform data augmentation on images in the input folder
    augment_images_in_folder(input_folder, output_folder, max_files)

    # Specify input and output folders
    input_folder = "data\VALIDATION\DANCE"
    output_folder = "data\VALIDATION\DANCE"
    max_files = 150
    # Perform data augmentation on images in the input folder
    augment_images_in_folder(input_folder, output_folder, max_files)

    # Specify input and output folders
    input_folder = "data\VALIDATION\DOWNDOG"
    output_folder = "data\VALIDATION\DOWNDOG"
    max_files = 150
    # Perform data augmentation on images in the input folder
    augment_images_in_folder(input_folder, output_folder, max_files)

