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
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import re
from utils import *


# set seeds
torch.manual_seed(231)
np.random.seed(231)

join = os.path.join


"""
        {
         'image': prepare_image(image_1, resize_transform, sam),
         'boxes': resize_transform.apply_boxes_torch(image_1_boxes, image_1.shape[:2]),
         'original_size': image_1.shape[:2]
     }

"""

def create_npz_dataset(data):
    # print("dir: ", os.listdir("dataset/MedSAMDemo_2D/test/images/"))
    data_path = f"dataset/{data}"
    img_names = os.listdir(f"{data_path}/images")
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    for i, img_name in enumerate(img_names):
        print(f"Processing image {i}")
        print(f"{img_name=}")
        # Since images are in grayscale, all three channels have the same value
        img = io.imread(f"{data_path}/images/{img_name}") # shape (img_width, img_height, 3) 
        label = io.imread(f"{data_path}/labels/{img_name}") # shape (img_width, img_height, 1)

        processed_img = prepare_image(img, resize_transform, sam)
        original_size = img.shape[:2]
        with torch.no_grad():
            img_embeddings = sam.image_encoder(preprocess(processed_img.unsqueeze(dim=0)))

        print(f"creatind dataset: {original_size=}, {processed_img.shape=}")


        img_num = int(re.findall(r'\d+', img_name)[0])

        np.savez(f'dataset/npz/{data}/{img_name}.npz', 
                image=processed_img.cpu(), 
                label=label, 
                original_size=original_size,
                img_embeddings=img_embeddings.cpu(),
                img_num=img_num)    

        # if i == 1: break





# create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]

        # max_length = max(len(d['label']) for d in self.npz_data)
        # # Pad all arrays with zeros using NumPy
        # padded_labels = [np.pad(d['label'], (0, max_length - len(d['label'])), 'constant') for d in self.npz_data]

        self.labels = [data['label'] for data in self.npz_data]
        self.images = [data['image'] for data in self.npz_data]
        self.original_sizes = [data['original_size'] for data in self.npz_data]
        self.embeddings = [data['img_embeddings'] for data in self.npz_data]
        self.img_nums = [data['img_num'] for data in self.npz_data]
        print(f"{self.images[0].shape=}")
        print(f"{len(self.images)=}, {len(self.labels)=}")
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # return self.npz_data[index]


        image = self.images[index]
        label = self.labels[index]
        bboxes = torch.tensor(np.array([get_bbox_from_mask(label)]))
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



def segment_img(data, file_name):

    # img_names = os.listdir(npz_file)

    
    # data = np.load("embeddings/malignant_26.png.npz")
    # img_emb = data['img_emb']
    # img_label = data['label']
    # print(img_emb.shape)
    # print(img_label.shape)

    # print("image_name: ", img_name)

    dataset = NpzDataset(data)
    data_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # image_1 = io.imread("dataset/malignant/images/malignant_102.png")
    # image_2 = io.imread("dataset/malignant/images/malignant_103.png")

    # label_1 = io.imread("dataset/malignant/labels/malignant_102.png")
    # label_2 = io.imread("dataset/malignant/labels/malignant_103.png")

    # image_1_boxes = torch.tensor(np.array([get_bbox_from_mask(label_1)]))
    # image_2_boxes = torch.tensor(np.array([get_bbox_from_mask(label_2)]))

    # batched_input = [
    #  {
    #      'image': prepare_image(image_1, resize_transform, sam),
    #      'boxes': resize_transform.apply_boxes_torch(image_1_boxes, image_1.shape[:2]),
    #      'original_size': image_1.shape[:2]
    #  },
    #  {
    #      'image': prepare_image(image_2, resize_transform, sam),
    #      'boxes': resize_transform.apply_boxes_torch(image_2_boxes, image_2.shape[:2]),
    #      'original_size': image_2.shape[:2]
    #  }
    # ]

    for i, batch in enumerate(data_dataloader):

        # print(f'{batch['image'].shape}=')
        # print(batch)

        print(f"{len(batch)=}")

        batched_output = sam([batch], multimask_output=False)

        # print("batched_output[0].keys(): ", batched_output[0].keys())

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        img_num = batch['img_num'].item()
        image_1 = io.imread(f"dataset/malignant/images/{file_name}_{img_num}.png")
        label_1 = io.imread(f"dataset/malignant/labels/{file_name}_{img_num}.png")
        image_1_boxes = torch.tensor(np.array([get_bbox_from_mask(label_1)]))



        ax.imshow(image_1)
        for mask in batched_output[0]['masks']:
            show_mask(mask.cpu().numpy(), ax, random_color=True)
        for box in image_1_boxes:
            show_box(box.cpu().numpy(), ax)
        ax.axis('off')

        # ax[1].imshow(image_2)
        # for mask in batched_output[1]['masks']:
        #     show_mask(mask.cpu().numpy(), ax[1], random_color=True)
        # for box in image_2_boxes:
        #     show_box(box.cpu().numpy(), ax[1])
        # ax[1].axis('off')

        plt.tight_layout()
        plt.show()
        plt.savefig(f"batched_{i}.png")



    # predictor.set_image(image)

    # input_point = np.array([[325, 32]])
    # input_label = np.array([1])

    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()

    # masks, scores, logits = predictor.predict(
    #     point_coords=None,
    #     box=bboxes,
    #     multimask_output=True,
    # )

    # show_masks_on_image(image, masks, scores)

    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(image),
    #     show_masks_on_image(image, mask, score),
    #     # show_mask(mask, plt.gca())
    #     # show_points(input_point, input_label, plt.gca())
    #     # show_boxes_on_image(image, [bboxes])
    #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.savefig(f"output_{i}.png")





        # break

def preprocess(x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - sam.pixel_mean) / sam.pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = sam.image_encoder.img_size - h
    padw = sam.image_encoder.img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    print(f"{x.shape=}")
    return x



if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    # processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    # malignant_dataset()

    # masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())

    # NpzDataset("embeddings")

    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    # predictor = SamPredictor(sam)
    # create_npz_dataset('benign')
    # create_npz_dataset('malignant')
    
    # NpzDataset('dataset/npz/malignant')
    segment_img('dataset/npz/malignant', 'malignant')
    
    

