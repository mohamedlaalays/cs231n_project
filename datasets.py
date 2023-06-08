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
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import re
from utils import *
from experiment_setup import*

# set seeds
torch.manual_seed(231)
np.random.seed(231)

join = os.path.join

sam_checkpoint = "models/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)



def create_npz_dataset(data):
    data_path = f"dataset/{data}"
    img_names = os.listdir(f"{data_path}/images")
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    for i, img_name in enumerate(img_names):
        print(f"Processing image ... {i}")
        # Since images are in grayscale, all three channels have the same value
        img = io.imread(f"{data_path}/images/{img_name}") # shape (img_width, img_height, 3) 
        label = io.imread(f"{data_path}/labels/{img_name}") # shape (img_width, img_height, 1)

        processed_img = prepare_image(img, resize_transform, sam)
        original_size = img.shape[:2]
        with torch.no_grad():
            img_embeddings = sam.image_encoder(preprocess(processed_img.unsqueeze(dim=0)))

        img_num = int(re.findall(r'\d+', img_name)[0])

        np.savez(f'dataset/npz_base/{data}/{img_name}.npz', 
                image=processed_img.cpu(), 
                label=label, 
                original_size=original_size,
                img_embeddings=img_embeddings.cpu(),
                img_num=img_num)    

        # if i == 210: break





# create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root, bbox_size):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(self.data_root, f)) for f in self.npz_files]
        self.bbox_size = bbox_size

        # max_length = max(len(d['label']) for d in self.npz_data)
        # # Pad all arrays with zeros using NumPy
        # padded_labels = [np.pad(d['label'], (0, max_length - len(d['label'])), 'constant') for d in self.npz_data]

        self.labels = [data['label'] for data in self.npz_data]
        self.images = [data['image'] for data in self.npz_data]
        self.original_sizes = [data['original_size'] for data in self.npz_data]
        self.embeddings = [data['img_embeddings'] for data in self.npz_data]
        self.img_nums = [data['img_num'] for data in self.npz_data]
        print(f"loaded {len(self.images)} images from {data_root}")
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        bboxes = torch.tensor(np.array([get_bbox_from_mask(label, self.bbox_size)]))
        original_size = self.original_sizes[index]
        img_embeddings = self.embeddings[index]
        img_num = self.img_nums[index]

        return {
         'image': image,
         'boxes': resize_transform.apply_boxes_torch(bboxes, original_size),
         'original_size': original_size,
         'img_embeddings': img_embeddings,
         'img_num': img_num # Apparently dataloader doesn't like strings
        }
    


    # def collate_fn(self, data): 

    #     # image = self.images[index]
    #     # label = self.labels[index]
    #     # bboxes = torch.tensor(np.array([get_bbox_from_mask(label)]))
    #     # original_size = self.original_sizes[index]
        
    #     # return {
    #     #  'image': image,
    #     # #  'boxes': resize_transform.apply_boxes_torch(bboxes, original_size),
    #     #  'original_size': original_size
    #     # }

    #     images = [torch.tensor(d['image']) for d in data] #(3)
    #     labels = [d['label'] for d in data]
    #     original_sizes = [d['original_size'] for d in data]

    #     images = pad_sequence(images, batch_first=True) #(4)
    #     # labels = torch.tensor(labels) #(5)

    #     return { #(6)
    #         'image': images,
    #         'boxes': torch.tensor(np.array([get_bbox_from_mask(label)])),
    #         'original_size': original_size
    #         # 'label': labels
    #     }


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()




def preprocess(x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - sam.pixel_mean) / sam.pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = sam.image_encoder.img_size - h
    padw = sam.image_encoder.img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x



# if __name__ == "__main__":

    # create_npz_dataset('malignant')
    # create_npz_dataset('benign')
    




    
    

