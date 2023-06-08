import matplotlib.pyplot as plt
import json

def graph_result(result_name, dataset):
    f = open(f'results/{result_name}.json')
    data = json.load(f)
    x_values = []
    y_values_dice = []
    y_values_iou = []

    for key, value in data.items():
        x_values.append(int(key))
        y_values_dice.append(value["avg_dice_coef"])
        y_values_iou.append(value["avg_iou_score"])

    # Plotting the graph
    plt.plot(x_values, y_values_dice, label="Average Dice Coefficient")
    plt.plot(x_values, y_values_iou, label="Average IOU Score")
    plt.xlabel("Bounding box increase (pixels)")
    plt.ylabel("Score")
    plt.title(f"Graph of Average Dice Coefficient and Average IOU Score ({dataset})")
    plt.legend()
    # plt.show()
    plt.savefig(f'results/{result_name}.png')


if __name__ == "__main__":
    graph_result("malignant_vit_h_bbox", "Malignant")