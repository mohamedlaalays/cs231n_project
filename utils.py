import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.color import gray2rgb
from skimage.util import img_as_ubyte

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.savefig("random_pts.png")
    plt.show()

# def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
#     plt.figure(figsize=(10,10))
#     plt.imshow(raw_image)
#     input_points = np.array(input_points)
#     if input_labels is None:
#       labels = np.ones_like(input_points[:, 0])
#     else:
#       labels = np.array(input_labels)
#     show_points(input_points, labels, plt.gca())
#     for box in boxes:
#       show_box(box, plt.gca())
#     plt.axis('on')
#     plt.show()


def show_points_and_boxes_gt_on_image(superposed_image_path, boxes, input_points, input_labels):
    raw_image = io.imread(superposed_image_path)
    
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())


    plt.savefig(superposed_image_path)
    plt.axis('on')
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
      # mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    plt.savefig(f"output.png")
    plt.show()


def get_bbox_from_mask(mask, dist):
    '''Returns a bounding box from a mask'''
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    # x_min = max(0, x_min - np.random.randint(0, 20))
    # x_max = min(W, x_max + np.random.randint(0, 20))
    # y_min = max(0, y_min - np.random.randint(0, 20))
    # y_max = min(H, y_max + np.random.randint(0, 20))
    x_min = max(0, x_min - dist)
    x_max = min(W, x_max + dist)
    y_min = max(0, y_min - dist)
    y_max = min(H, y_max + dist)

    return np.array([x_min, y_min, x_max, y_max])


def superpose_img_label(original_image, segmentation_mask, img_num):
  original_image = cv2.imread(original_image)
  segmentation_mask = cv2.imread(segmentation_mask, 0)  # Load as grayscale
  # Create a colored mask by converting the grayscale mask to color
  colored_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2BGR)
  # Set the color of the mask (e.g., green)
  mask_color = (0, 255, 0)  # Green color
  # Apply the mask color to the colored mask
  colored_mask[np.where((colored_mask == [255, 255, 255]).all(axis=2))] = mask_color
  # Superpose the colored mask on the original image
  superposed_image = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
  # print(f"{superposed_image}")
  # img_arr = superposed_image
  path = f'sample_images/org_label_{img_num}.png'
  cv2.imwrite(path, superposed_image)
  return path



def superpose_img_mask(img_path, label_path, mask, img_num, dist):
  fig, ax = plt.subplots()
  image = io.imread(img_path)
  label = io.imread(label_path)
  image_box = get_bbox_from_mask(label, dist)
  ax.imshow(image, aspect='auto')
  show_mask(mask, ax, random_color=True)
  show_box(image_box, ax)
  ax.axis('off')

  plt.tight_layout()
  plt.savefig(f"sample_images/pred_label_{img_num}.png")



# def side_by_side(original_image_path, segmentation_mask_path, img_num):
#   original_image = io.imread(original_image_path)
#   segmentation_mask = io.imread(segmentation_mask_path)
#   # Convert the segmentation mask to RGB if it is grayscale
#   if len(segmentation_mask.shape) == 2:
#       segmentation_mask = gray2rgb(segmentation_mask)
#   # Normalize the segmentation mask values to [0, 1]
#   segmentation_mask = segmentation_mask.astype(np.float32) / 255.0
#   # Create a copy of the original image
#   output_image = np.copy(original_image)
#   # Superimpose the segmentation mask on the original image
#   alpha = 0.7  # Controls the transparency of the segmentation mask
#   output_image = (output_image * (1 - alpha) + segmentation_mask * alpha).astype(np.uint8)

#   io.imsave(f'sample_images/org_label_{img_num}.png', img_as_ubyte(output_image))