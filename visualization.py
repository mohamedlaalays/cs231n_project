import matplotlib.pyplot as plt
import json

def graph_result(result_name, result_name_two, dataset, dataset_two, file_name):
    f = open(f'results/{result_name}.json')
    data = json.load(f)
    x_values = []
    y_values_dice = []
    y_values_iou = []

    for key, value in data.items():
        x_values.append(int(key))
        y_values_dice.append(value["avg_dice_coef"])
        y_values_iou.append(value["avg_iou_score"])


    f = open(f'results/{result_name_two}.json')
    data = json.load(f)
    x_values_two = []
    y_values_dice_two = []
    y_values_iou_two = []

    for key, value in data.items():
        x_values_two.append(int(key))
        y_values_dice_two.append(value["avg_dice_coef"])
        y_values_iou_two.append(value["avg_iou_score"])


    # Plotting the graph
    plt.plot(x_values, y_values_dice, label=f"Average Dice {dataset}")
    plt.plot(x_values, y_values_iou, label=f"Average IoU {dataset}")
    plt.plot(x_values_two, y_values_dice_two, label=f"Average Dice {dataset_two}")
    plt.plot(x_values_two, y_values_iou_two, label=f"Average IoU {dataset_two}")
    plt.xlabel("Bounding Box Increase (Pixels)")
    plt.ylabel("Score")
    # plt.ylim(0, 1)
    plt.title(f"Bounding box against average Dice and IoU")
    plt.legend()
    # plt.show()
    plt.savefig(f'results/{file_name}.png')


if __name__ == "__main__":
    # graph_result("benign_vit_h_num_pts", "Benign")
    graph_result("benign_vit_h_bbox", "malignant_vit_h_bbox", "Benign", "Malignant", "bbox")