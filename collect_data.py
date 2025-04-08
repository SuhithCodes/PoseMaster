import os
import requests

def download_images_from_files(input_dir):
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_dir, file_name)
            output_dir = os.path.splitext(file_path)[0] + "_images"
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        # Split the line into image filename and URL
                        filename, url = line.strip().split('\t')

                        # Create directory if it doesn't exist
                        os.makedirs(os.path.join(output_dir, os.path.dirname(filename)), exist_ok=True)

                        # Download the image
                        response = requests.get(url)
                        if response.status_code == 200:
                            with open(os.path.join(output_dir, filename), 'wb') as img_file:
                                img_file.write(response.content)
                            print(f"Downloaded {filename} successfully.")
                        else:
                            print(f"Failed to download {filename}. Status code: {response.status_code}")

                    except Exception as e:
                        print(f"Error processing line: {line}\nError: {e}")

# Example usage
input_dir = "data\Yoga-82\yoga_dataset_links"
download_images_from_files(input_dir)
