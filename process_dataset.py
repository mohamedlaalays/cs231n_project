import os
import shutil

# Specify the path to the Dataset folder
dataset_folder = "dataset/Dataset_BUSI_with_GT"

# Create the labels folder
labels_folder = os.path.join(dataset_folder, "labels")
os.makedirs(labels_folder, exist_ok=True)

# Iterate over the subfolders in the Dataset folder
for subfolder_name in os.listdir(dataset_folder):
    subfolder_path = os.path.join(dataset_folder, subfolder_name)
    if not os.path.isdir(subfolder_path):
        continue  # Skip non-directory files
    
    # Create the subfolder labels folder
    subfolder_labels_folder = os.path.join(labels_folder, subfolder_name + "labels")
    os.makedirs(subfolder_labels_folder, exist_ok=True)
    
    # Iterate over the files in the subfolder
    for filename in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, filename)
        if not os.path.isfile(file_path):
            continue  # Skip subdirectories
        
        if "_mask" in filename:
            # Move the file to the subfolder labels folder
            new_file_path = os.path.join(subfolder_labels_folder, filename)
            shutil.move(file_path, new_file_path)
