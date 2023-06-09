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


# def calculate_absolute_coordinates(part_index, point_in_part, image_size):
#     part_size = (image_size[0] // 2, image_size[1] // 2)

#     part_row = part_index // 2
#     part_col = part_index % 2

#     absolute_x = part_col * part_size[1] + point_in_part[0]
#     absolute_y = part_row * part_size[0] + point_in_part[1]

#     return np.array([absolute_x, absolute_y])



# def get_corner_pts(raw_label, bbox, num_pts):

#     x_min, y_min, x_max, y_max = bbox
#     box_region = raw_label[y_min:y_max, x_min:x_max]

#     height, width = box_region.shape
#     split_height = height // 2
#     split_width = width // 2

#     q1 = np.copy(box_region[:split_height, :split_width])
#     q2 = box_region[:split_height, split_width:]
#     q3 = box_region[split_height:, :split_width]
#     q4 = box_region[split_height:, split_width:]

#     # for i in range(num_pts):
#     print(f'{raw_label.shape=}')
#     print(f'{box_region.shape=}')

#     # return get_random_pts(q1, (0, 0, q1.shape[0], q1.shape[1]), num_pts)
#     # print(f'{q4[0]=}')
#     # print(f'{q1[-1]=}')
#     point, label = get_random_pts(q1, (0, 0, q1.shape[0], q1.shape[1]), num_pts)
#     # point, label = get_random_fg(q4, num_pts)
#     print(f'{point[0]=}')
#     abs_pts = calculate_absolute_coordinates(1, point[0], raw_label.shape)
#     # abs_pts = calculate_absolute_coordinates(4, point[0], raw_label.shape)
#     print(f'{abs_pts=}')
#     return [abs_pts], label
     

def get_corner_points(raw_label, bbox):
    # print(f'{raw_label.shape=}')
    x_min, y_min, x_max, y_max = bbox
    # width = x_max - x_min
    # height = y_max - y_min
    # part_width = width // 2
    # part_height = height // 2
    # bottom_right_x_offset = np.random.randint(part_width, width)
    # bottom_right_y_offset = np.random.randint(part_height, height)
    # point_x = x_min + bottom_right_x_offset
    # point_y = y_min + bottom_right_y_offset
    # return np.array([[point_x, point_y]])

    width = x_max - x_min
    height = y_max - y_min

    # Divide the width and height by 2 to get the size of each divided part
    part_width = width // 2
    part_height = height // 2

    # Generate random offsets for each part
    x_offsets = np.random.randint(0, part_width, size=4)
    y_offsets = np.random.randint(0, part_height, size=4)

    # Calculate the coordinates of the chosen points with respect to the grayscale image
    points = []
    labels = []
    for i in range(4):
        point_x = x_min + (part_width * (i % 2)) + x_offsets[i]
        point_y = y_min + (part_height * (i // 2)) + y_offsets[i]
        labels.append(raw_label[point_y, point_x])
        # print(f'{raw_label[point_y, point_x]=}')
        points.append([point_x, point_y])

    # print(f'{raw_label[points[0][0]]}')
    return np.array(points), np.array(labels)





        



if __name__ == "__main__":
    raw_label = io.imread(f"dataset/original/benign/labels/benign_5.png", as_gray=True)
    # print(f'{raw_label.shape=}')
    # together = raw_image + raw_label
    bbox = get_bbox_from_mask(raw_label, 40)
    # input_points, input_labels = get_random_bg(raw_label, bbox, 1000)
    # input_points, input_labels = get_random_fg(raw_label, 1000)
    # input_points, input_labels = get_random_pts(raw_label, bbox, 3)
    # fg_count = 0
    # for i in range(1000):
    #     input_points, input_labels = get_random_pts(raw_label, bbox, 3000)
    #     fg_count += np.count_nonzero(input_labels)
    # print(f'foreground points: {fg_count/1000}')
    # input_points, input_labels = get_corner_pts(raw_label, bbox, 1)
    input_points, input_labels = get_corner_points(raw_label, bbox)
    # input_labels = [1, 1, 1, 1]
    show_points_on_image(raw_label, input_points, input_labels=input_labels)