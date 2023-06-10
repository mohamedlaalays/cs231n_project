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

    labels = np.ones(num_pts)

    return random_points, labels


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

    labels = np.zeros(num_pts)

    return random_points, labels


def get_random_pts(raw_label, bbox, num_pts):
    # bg_pts = random.randint(1, num_pts+1)

    fg_pts, bg_pts = [], []
    fg_labels, bg_labels = [], []
    val =  random.randint(1, num_pts)
    if np.random.random_sample() < 0.5:
        bg_pts,  bg_labels= get_random_bg(raw_label, bbox, val)
        fg_pts, fg_labels = get_random_fg(raw_label, num_pts-val)
    else:
        bg_pts, bg_labels= get_random_bg(raw_label, bbox, num_pts-val)
        fg_pts, fg_labels = get_random_fg(raw_label, val)

    return np.concatenate((fg_pts, bg_pts)), np.concatenate((fg_labels, bg_labels)) 


if __name__ == "__main__":
    raw_label = io.imread(f"dataset/original/benign/labels/benign_5.png", as_gray=True)
    print(f'{raw_label.shape=}')
    # together = raw_image + raw_label
    bbox = get_bbox_from_mask(raw_label, 200)
    # input_points, input_labels = get_random_bg(raw_label, bbox, 1000)
    input_points, input_labels = get_random_fg(raw_label, 1000)
    # input_points, input_labels = get_random_pts(raw_label, bbox, 3)
    # fg_count = 0
    # for i in range(1000):
    #     input_points, input_labels = get_random_pts(raw_label, bbox, 3000)
    #     fg_count += np.count_nonzero(input_labels)
    # print(f'foreground points: {fg_count/1000}')
    show_points_on_image(raw_label, input_points, input_labels=input_labels)