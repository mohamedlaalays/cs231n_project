import numpy as np
import random
from skimage import io
from utils import*


def get_random_fg(image, num_pts):
    image = image.transpose(1, 0)
    foreground = np.argwhere(image == True)
    # Randomly select k white points from the indices
    random_indices = random.sample(range(len(foreground)), num_pts)
    random_points = foreground[random_indices]

    return random_points


def get_random_bg(image, bbox, num_pts):
    image = image.transpose(1, 0)
    x_min, y_min, x_max, y_max = bbox
    # Get the sub-image within the box region
    box_image = image[x_min:x_max, y_min:y_max]

    # Find the indices of white pixels within the box
    white_indices = np.argwhere(box_image == False)

    # Randomly select k white points from the indices
    random_indices = random.sample(range(len(white_indices)), num_pts)
    random_points = white_indices[random_indices]

    # Calculate the coordinates of the randomly selected white points within the original image
    random_points[:, 0] += x_min
    random_points[:, 1] += y_min

    return random_points


if __name__ == "__main__":
    raw_label = io.imread(f"dataset/original/benign/labels/benign_5.png", as_gray=True)
    print(f'{raw_label.shape=}')
    # together = raw_image + raw_label
    bbox = get_bbox_from_mask(raw_label)
    input_points = get_random_bg(raw_label, bbox, 1000)
    show_points_on_image(raw_label, input_points, input_labels=None)