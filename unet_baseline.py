import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D

def load_dataset(dataset_path, target_size):
    benign_images = []
    benign_labels = []
    malignant_images = []
    malignant_labels = []

    for filename in os.listdir(os.path.join(dataset_path, 'benign', 'images')):
        img_path = os.path.join(dataset_path, 'benign', 'images', filename)
        label_path = os.path.join(dataset_path, 'benign', 'labels', filename)

        # Read and resize image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0

        # Read and resize label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, target_size)
        label = label.astype('float32') / 255.0

        benign_images.append(img)
        benign_labels.append(label)

    for filename in os.listdir(os.path.join(dataset_path, 'malignant', 'images')):
        img_path = os.path.join(dataset_path, 'malignant', 'images', filename)
        label_path = os.path.join(dataset_path, 'malignant', 'labels', filename)

        # Read and resize image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0

        # Read and resize label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, target_size)
        label = label.astype('float32') / 255.0

        malignant_images.append(img)
        malignant_labels.append(label)

    # Convert lists to numpy arrays
    benign_images = np.array(benign_images)
    benign_labels = np.array(benign_labels)
    malignant_images = np.array(malignant_images)
    malignant_labels = np.array(malignant_labels)

    return benign_images, benign_labels, malignant_images, malignant_labels


def preprocess_dataset(benign_images, benign_labels, malignant_images, malignant_labels):
    # Reshape images to have a single channel
    benign_images = np.expand_dims(benign_images, axis=-1)
    benign_labels = np.expand_dims(benign_labels, axis=-1)
    malignant_images = np.expand_dims(malignant_images, axis=-1)
    malignant_labels = np.expand_dims(malignant_labels, axis=-1)

    return benign_images, benign_labels, malignant_images, malignant_labels



# Define the loss function (dice coefficient)
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    dice = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)
    return dice


def iou_score(y_true, y_pred):
    smooth = 1.0
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Load the dataset
dataset_path = 'dataset'
# benign_images, benign_labels, malignant_images, malignant_labels = load_dataset(dataset_path)

# Load the dataset
target_size = (256, 256)
benign_images, benign_labels, malignant_images, malignant_labels = load_dataset(dataset_path, target_size)

# Load the dataset
dataset_path = 'dataset'
# benign_images, benign_labels, malignant_images, malignant_labels = load_dataset(dataset_path)

# Load the dataset
target_size = (256, 256)
benign_images, benign_labels, malignant_images, malignant_labels = load_dataset(dataset_path, target_size)

import matplotlib.pyplot as plt

# Assuming you want to visualize the first image in the 'benign_images' array
image_index = 1
image = benign_images[image_index]
print(image.shape)

# Display the image
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

# Preprocess the dataset
benign_images, benign_labels, malignant_images, malignant_labels = preprocess_dataset(
    benign_images, benign_labels, malignant_images, malignant_labels)

image_index = 1
image = benign_images[image_index]
print(image.shape)

# Display the image
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

print(benign_images.dtype)

# Concatenate the benign and malignant data
all_images = np.concatenate((benign_images, malignant_images), axis=0)
all_labels = np.concatenate((benign_labels, malignant_labels), axis=0)

# Shuffle the data
indices = np.arange(all_images.shape[0])
np.random.shuffle(indices)
all_images = all_images[indices]
all_labels = all_labels[indices]

from sklearn.model_selection import train_test_split
# Split the data into training, validation, and test sets
train_images, val_test_images, train_labels, val_test_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(val_test_images, val_test_labels, test_size=0.5, random_state=42)

# Define the U-Net model architecture
def unet_model(input_shape):
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    # Expanding path
    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define other training parameters (learning rate, batch size, etc.)
learning_rate = 0.0001
batch_size = 32
epochs = 100


# Create an instance of the U-Net model
input_shape = (train_images.shape[1], train_images.shape[2], train_images.shape[3])

# Reshape train_images to remove the extra dimension
#train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2])

# Update the input_shape based on the new shape of train_images
#input_shape = train_images[0].shape + (1,)

# Create an instance of the U-Net model
model = unet_model(input_shape)

import pickle

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=[dice_coefficient])

# Train the model
history_callback = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(val_images, val_labels))

# Save the training history
with open('history.pickle', 'wb') as file:
    pickle.dump(history_callback.history, file)

# Save the trained model
model.save('history_model.h5')

# Load the saved history
with open('history.pickle', 'rb') as file:
    saved_history = pickle.load(file)

import matplotlib.pyplot as plt

import pickle
import matplotlib.pyplot as plt

# Load the saved history
with open('history.pickle', 'rb') as file:
    saved_history = pickle.load(file)

# Plot the loss curves
plt.plot(saved_history['loss'], label='Training Loss')
plt.plot(saved_history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the accuracy curves
plt.plot(saved_history['dice_coefficient'], label='Training Accuracy')
plt.plot(saved_history['val_dice_coefficient'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Dice coeddicient')
plt.legend()
plt.show()





#visualize outputs
#given image w mask, run model on that image and take the segmented masks and visualize that with the ground truth masks
#other baselines:
#stuff we've done in class like taking otsu thrshold and what happens then
#find pixel intensity value that best slits the data into 2, binarize pixel values based on that and will get a mask

#image aug validation
#

import random

# Randomly select 3 benign images and labels
test_sample_indices = random.sample(range(len(test_images)), 3)
test_samples = [test_images[i] for i in test_sample_indices]
test_labels_samples = [test_labels[i] for i in test_sample_indices]

# Print the selected images and labels
# print("Benign Samples:")
# for i in range(3):
#     print("Image:", benign_samples[i])
#     print("Label:", benign_labels_samples[i])

test_image_1 = test_samples[0]
test_image_2 = test_samples[1]
test_image_3 = test_samples[2]

fig, axs = plt.subplots(1, 2)

#train_images, train_labels
# Display the main image
axs[0].imshow(train_images[0], cmap='gray')
axs[0].axis('off')

# Display the label image
axs[1].imshow(train_labels[0], cmap='gray')
axs[1].axis('off')

# Set the spacing between subplots
plt.subplots_adjust(wspace=0.2)

# Show the plot
plt.show()

import matplotlib.pyplot as plt

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2)

# Display the main image
axs[0].imshow(test_image_1, cmap='gray')
axs[0].axis('off')

# Display the label image
axs[1].imshow(test_label_1, cmap='gray')
axs[1].axis('off')

# Set the spacing between subplots
plt.subplots_adjust(wspace=0.2)

# Show the plot
plt.show()

predictions = model.predict(np.expand_dims(train_images[0], axis=0))

binary_predictions = (predictions > 0.5).astype(np.uint8)  # Convert probabilities to binary values
segmentation_mask = np.squeeze(binary_predictions)  # Remove the batch dimension if present

import matplotlib.pyplot as plt

# Load and preprocess the original image
original_image = train_images[0] # Load the original image without preprocessing

# Plot the original image, segmentation mask, and original segmented image label side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot the original image
axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

# Plot the segmentation mask
axes[1].imshow(segmentation_mask, cmap='gray')
axes[1].set_title('Segmentation Mask')
axes[1].axis('off')

# Plot the original segmented image label
axes[2].imshow(train_labels[0], cmap='gray')
axes[2].set_title('Original Segmented Image Label')
axes[2].axis('off')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Load and preprocess the original image
original_image = test_images[0] # Load the original image without preprocessing

# Plot the original image, segmentation mask, and original segmented image label side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot the original image
axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

# Plot the segmentation mask
axes[1].imshow(segmentation_mask, cmap='gray')
axes[1].set_title('Segmentation Mask')
axes[1].axis('off')

# Plot the original segmented image label
axes[2].imshow(train_labels[0], cmap='gray')
axes[2].set_title('Original Segmented Image Label')
axes[2].axis('off')

plt.tight_layout()
plt.show()

#resnet encoder
#nn.unet