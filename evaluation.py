import torch
from transformers import SamModel, SamProcessor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from utils import *
import os


"""
1--> get datasets with segmentation label
2--> create dataset embeddings
3--> bounding box, foreground, and background
4--> 

"""




def sam_iou():
    """
    
    """

    # data = np.load('my_arrays.npz')

    # # Retrieve the individual arrays by their names
    # img_emb = data['img_emb']
    # label = data['label']

    # print("img_emb.shape: ", img_emb.shape)
    # print("label.shape: ", label.shape)

    # print("lable: ", label)

    # plt.imshow(label)
    # plt.show()
    # return


    # print("dir: ", os.listdir("dataset/MedSAMDemo_2D/test/images/"))
    data_path = "dataset/MedSAMDemo_2D/test"
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