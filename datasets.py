# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
import torch
from transformers import SamModel, SamProcessor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from utils import *
import os

# set seeds
torch.manual_seed(231)
np.random.seed(231)


# create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
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



def sam_iou():


    # print("dir: ", os.listdir("dataset/MedSAMDemo_2D/test/images/"))
    data_path = "dataset/Dataset_BUSI_with_GT/malignant"
    img_names = os.listdir(f"{data_path}/images")
    for img_name in img_names:
        img = Image.open(f"{data_path}/images/{img_name}") # WHAT IS THE SHAPE OF THE RETURNED IMG
        label = Image.open(f"{data_path}/labels/{img_name}")

        inputs = processor(img, return_tensors="pt").to(device)

        print("inputs: ", inputs.keys())

        img_emb = model.get_image_embeddings(inputs["pixel_values"])

        np.savez(f'{data_path}/processed/{img_name}.npz', img_emb=img_emb, label=label)

        break


    # img = "dataset/MedSAMDemo_2D/test/images/FLARE22_Tr_000400.png"
    # label = "dataset/MedSAMDemo_2D/test/labels/FLARE22_Tr_000400.png"
    # raw_image = np.array(Image.open(img)) # convert image to black and white
    # raw_image_2 = Image.open(requests.get(img, stream=True).raw).convert("RGB")

    # print("raw_image.dtype: ", raw_image[200])
    # raw_image.show()

    # plt.imshow(raw_image)
    
    # inputs = processor(raw_image, return_tensors="pt").to(device)
    # print("inputs: ", inputs['pixel_values'].shape)

    # image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
    # print("image_embeddings: ", image_embeddings.shape)

    # input_points = [[[450, 600]]]
    # show_points_on_image(raw_image, input_points[0])    

    # plt.show()

    # inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    # # pop the pixel_values as they are not neded
    # inputs.pop("pixel_values", None)
    # inputs.update({"image_embeddings": image_embeddings})

    # with torch.no_grad():
    #     outputs = model(**inputs)

    # masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    # scores = outputs.iou_scores

    # show_masks_on_image(raw_image, masks[0], scores)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    sam_iou()