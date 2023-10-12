import os
import shutil

# # Specify the path to the Dataset folder
# dataset_folder = "dataset/Dataset_BUSI_with_GT"

# # Create the labels folder
# labels_folder = os.path.join(dataset_folder, "labels")
# os.makedirs(labels_folder, exist_ok=True)

# # Iterate over the subfolders in the Dataset folder
# for subfolder_name in os.listdir(dataset_folder):
#     subfolder_path = os.path.join(dataset_folder, subfolder_name)
#     if not os.path.isdir(subfolder_path):
#         continue  # Skip non-directory files
    
#     # Create the subfolder labels folder
#     subfolder_labels_folder = os.path.join(labels_folder, subfolder_name + "labels")
#     os.makedirs(subfolder_labels_folder, exist_ok=True)
    
#     # Iterate over the files in the subfolder
#     for filename in os.listdir(subfolder_path):
#         file_path = os.path.join(subfolder_path, filename)
#         if not os.path.isfile(file_path):
#             continue  # Skip subdirectories
        
#         if "_mask" in filename:
#             # Move the file to the subfolder labels folder
#             new_file_path = os.path.join(subfolder_labels_folder, filename)
#             shutil.move(file_path, new_file_path)



# # Specify the path to the dataset folder
# dataset_folder = "dataset"

# # Iterate over the subfolders ("benign" and "malignant")
# for subfolder in ["benign", "malignant"]:
#     subfolder_path = os.path.join(dataset_folder, subfolder)
    
#     # Iterate over the "images" and "labels" folders within each subfolder
#     for folder_name in ["images", "labels"]:
#         folder_path = os.path.join(subfolder_path, folder_name)
        
#         # Rename the files in the folder
#         for filename in os.listdir(folder_path):
#             current_file_path = os.path.join(folder_path, filename)
#             new_filename = filename.replace(" ", "_")
#             new_filename = new_filename.replace("(", "")
#             new_filename = new_filename.replace(")", "")
#             new_filename = new_filename.replace("_mask", "")
#             new_file_path = os.path.join(folder_path, new_filename)
            
#             # Rename the file
#             os.rename(current_file_path, new_file_path)


import os
import random


"""ChatGPT generated code
Prompt: i have a dataset folder. There are two subfolders: benign and malignant.\
      In each subfolder, there are more than 200 files. Write python code that \
        random selects 200 files in each subfolder and deletes the rest of files
"""

# # Set the path to the dataset folder
# dataset_path = 'dataset/npz_huge'

# # Set the number of files to keep in each subfolder
# num_files_to_keep = 200

# # Get the list of subfolders
# subfolders = ['benign', 'malignant']

# # Iterate over the subfolders
# for subfolder in subfolders:
#     # Get the path to the current subfolder
#     subfolder_path = os.path.join(dataset_path, subfolder)
    
#     # Get the list of files in the subfolder
#     files = os.listdir(subfolder_path)
    
#     # Check if the number of files is greater than the desired number to keep
#     if len(files) > num_files_to_keep:
#         # Randomly select files to keep
#         files_to_keep = random.sample(files, num_files_to_keep)
        
#         # Iterate over the files in the subfolder
#         for file in files:
#             # Get the path to the current file
#             file_path = os.path.join(subfolder_path, file)
            
#             # Check if the file should be deleted
#             if file not in files_to_keep:
#                 # Delete the file
#                 os.remove(file_path)
#                 print(f"Deleted file: {file_path}")



# Set the path to the dataset folder
# dataset_reference = 'dataset/npz_huge'

# Set the number of files to keep in each subfolder
# num_files_to_keep = 200

# # Get the list of subfolders
# subfolders = ['benign', 'malignant']

# dataset_interest = 'dataset/npz_base'

# # Iterate over the subfolders
# for subfolder in subfolders:
#     # Get the path to the current subfolder
#     ref_subfolder_path = os.path.join(dataset_reference, subfolder)
    
#     # Get the list of files in the subfolder
#     ref_files = os.listdir(ref_subfolder_path)

#     int_subfolder_path = os.path.join(dataset_interest, subfolder)
#     int_files = os.listdir(int_subfolder_path)

#     count = 0

#     for file in int_files:
#         file_path = os.path.join(int_subfolder_path, file)
#         if file in ref_files:
#             continue
#         else:
#             count += 1
#             os.remove(file_path)
#             print(f"Deleted file: {file_path}")
#             # break

#     print(f'Number of files deleted in {subfolder}: {count}')
    
    # # Check if the number of files is greater than the desired number to keep
    # if len(files) > num_files_to_keep:
    #     # Randomly select files to keep
    #     files_to_keep = random.sample(files, num_files_to_keep)
        
    #     # Iterate over the files in the subfolder
    #     for file in files:
    #         # Get the path to the current file
    #         file_path = os.path.join(subfolder_path, file)
            
    #         # Check if the file should be deleted
    #         if file not in files_to_keep:
    #             # Delete the file
    #             os.remove(file_path)
    #             print(f"Deleted file: {file_path}")