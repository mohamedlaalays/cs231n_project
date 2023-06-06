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
    # Normalize images
    #benign_images = [image.astype('float32') / 255.0 for image in benign_images]
    #malignant_images = [image.astype('float32') / 255.0 for image in malignant_images]

    # Reshape images to have a single channel
    benign_images = [np.expand_dims(image, axis=-1) for image in benign_images]
    benign_labels = [np.expand_dims(label, axis=-1) for label in benign_labels]
    malignant_images = [np.expand_dims(image, axis=-1) for image in malignant_images]
    malignant_labels = [np.expand_dims(label, axis=-1) for label in malignant_labels]

    # Convert lists to numpy arrays
    benign_images = np.array(benign_images)
    benign_labels = np.array(benign_labels)
    malignant_images = np.array(malignant_images)
    malignant_labels = np.array(malignant_labels)

    return benign_images, benign_labels, malignant_images, malignant_labels

    # Define the loss function (dice coefficient)
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    dice = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)
    return dice


# Load the dataset
dataset_path = 'dataset'
# benign_images, benign_labels, malignant_images, malignant_labels = load_dataset(dataset_path)

# Load the dataset
target_size = (256, 256)
benign_images, benign_labels, malignant_images, malignant_labels = load_dataset(dataset_path, target_size)

# Preprocess the dataset
benign_images, benign_labels, malignant_images, malignant_labels = preprocess_dataset(
    benign_images, benign_labels, malignant_images, malignant_labels)


# Concatenate the benign and malignant data
all_images = np.concatenate((benign_images, malignant_images), axis=0)
all_labels = np.concatenate((benign_labels, malignant_labels), axis=0)

# Shuffle the data
indices = np.arange(all_images.shape[0])
np.random.shuffle(indices)
all_images = all_images[indices]
all_labels = all_labels[indices]

# Split the data into training and validation sets
split = int(0.8 * all_images.shape[0])
train_images = all_images[:split]
train_labels = all_labels[:split]
val_images = all_images[split:]
val_labels = all_labels[split:]

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
learning_rate = 0.001
batch_size = 32
epochs = 100


# Create an instance of the U-Net model
input_shape = (train_images.shape[1], train_images.shape[2], train_images.shape[3])

# Create an instance of the U-Net model
model = unet_model(input_shape)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=[dice_coefficient])

# Train the model
model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(val_images, val_labels))

# Save the trained model
model.save('unet_model.h5')
