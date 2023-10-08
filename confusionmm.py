# This script performs the evaluation of predicted results by comparing them with ground truth results.
# It calculates the confusion matrix and plots a normalized confusion matrix using matplotlib.

import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def parse_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    results = [list(map(float, line.strip().split())) for line in lines]
    print(f"Length of results in {file_path}: {len(results)}")
    return np.array(results)

def calculate_confusion_matrix(predicted_results, true_results, num_classes):
    pred_classes = predicted_results[:, 0].astype(int)
    true_classes = true_results[:, 0].astype(int)

    # Ensure the classes are in the range [1, num_classes]
    pred_classes = np.clip(pred_classes, 1, num_classes)
    true_classes = np.clip(true_classes, 1, num_classes)
    cm = confusion_matrix(true_classes, pred_classes, labels=np.arange(1, num_classes + 1))
    return cm

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")

    fig.tight_layout()
    plt.savefig('confusion_matrix_gam_norm.png')  # Save the normalized confusion matrix as an image
    plt.show()

def main():
    test_dir = '/home/whut4/zwj/draw_confusion/gam/labels'  # Update with the directory containing the predicted results
    ground_truth_dir = '/home/whut4/zwj/draw_confusion/gt/labels'  # Update with the directory containing the ground truth results
    num_classes = 30  # Update with the number of classes in your data

    cm_total = np.zeros((num_classes, num_classes), dtype=float)

    for filename in os.listdir(test_dir):
        if filename.endswith('.txt'):
            test_file_path = os.path.join(test_dir, filename)
            ground_truth_file_path = os.path.join(ground_truth_dir, filename)

            predicted_results = parse_results(test_file_path)
            true_results = parse_results(ground_truth_file_path)
            if len(predicted_results) != len(true_results):
                continue
            cm = calculate_confusion_matrix(predicted_results, true_results, num_classes)
            cm_total += cm

    # Normalize the confusion matrix
    row_sums = cm_total.sum(axis=1, keepdims=True)
    cm_normalized = cm_total / row_sums
    # Format the normalized confusion matrix with two decimal places
    cm_normalized = np.round(cm_normalized, 2)

    print("Confusion Matrix:")
    print(cm_normalized)

    class_names = [str(i) for i in range(1, num_classes + 1)]
    plot_confusion_matrix(cm_normalized, class_names)

if __name__ == "__main__":
    main()
