# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
from PIL import Image
from skimage import transform, io, segmentation
from segment_anything import sam_model_registry, SamPredictor
from utils import *


# set seeds
torch.manual_seed(231)
np.random.seed(231)

join = os.path.join


# create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]

        max_length = max(len(d['label']) for d in self.npz_data)
        # Pad all arrays with zeros using NumPy
        padded_labels = [np.pad(d['label'], (0, max_length - len(d['label'])), 'constant') for d in self.npz_data]

        self.ori_gts = np.vstack([label for label in padded_labels])
        self.img_embeddings = np.vstack([d['img_emb'] for d in self.npz_data])
        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float()



def malignant_dataset():


    # print("dir: ", os.listdir("dataset/MedSAMDemo_2D/test/images/"))
    data_path = "dataset/malignant"
    img_names = os.listdir(f"{data_path}/images")
    for i, img_name in enumerate(img_names):
        # Since images are in grayscale, all three channels have the same value
        img = io.imread(f"{data_path}/images/{img_name}") # shape (img_width, img_height, 3) 
        label = io.imread(f"{data_path}/labels/{img_name}") # shape (img_width, img_height, 1)

        print(label.shape)
        print(img.shape)

        # outputs is dictionary that stores ['pixel_values', 'original_sizes', 'reshaped_input_sizes']
        img_outputs = processor(img, return_tensors="pt").to(device)    # ===> DOUBLE CHECK WHY THREE CHANNELS DO NOT HAVE THE SAME VALUES????????????
        # label_outputs = processor(img, return_tensors='pt').to(device)  # ===> DOUBLE CHECK WHY THREE CHANNELS DO NOT HAVE THE SAME VALUES????????????

        # print("inputs: ", label_outputs.keys())
        # print("inputs['original_sizes']: ", label_outputs['original_sizes'])
        # print("inputs['reshaped_input_sizes'].shape: ", label_outputs['reshaped_input_sizes'].shape)
        # print('inputs["pixel_values"]: ', label_outputs["pixel_values"].shape)
        # print("slice through one channel: ", img_outputs["pixel_values"][0, 0, 100, 100], img_outputs["pixel_values"][0, 1, 100, 100], img_outputs["pixel_values"][0, 2, 100, 100])
        # print("img_outputs['pixel_values']: ", img_outputs["pixel_values"])

        img_emb = model.get_image_embeddings(img_outputs["pixel_values"]) # (1, 256, 64, 64)
        # print(img_emb.shape)

        np.savez(f'embeddings/{img_name}.npz', img_emb=img_emb, label=label)    

        if i == 0: break


def segment_img(npz_file):

    img_names = os.listdir(npz_file)

    for i, img_name in enumerate(img_names):
        # print("image_name: ", img_name)
        data = np.load(f'{npz_file}/{img_name}')
        img_emb = data['img_emb']
        img_label = data['label']
        # print(img_emb.shape)
        # print(img_label.shape)

        image = io.imread("dataset/malignant/malignant_161.png")

        predictor.set_image(image)

        input_point = np.array([[500, 375]])
        input_label = np.array([1])

        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        plt.show()

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()





        break



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    # malignant_dataset()

    # masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())

    # NpzDataset("embeddings")

    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    segment_img('embeddings')

