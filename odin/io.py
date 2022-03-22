import matplotlib.pyplot as plt
from matplotlib import patches
import shutil
import torch
import numpy as np


def deboxing(bbox):
    corner = [bbox[0], bbox[1]]
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    return corner, height, width


def show_each_image(sample, boxes=None, pred_boxes=None):
    """
    plot the images with or without the bounding boxes
    """
    plt.figure(figsize=(16, 16))
    plt.imshow(sample[..., 0], plt.cm.gray)
    ax = plt.gca()
    if boxes is not None:
        for bbox in boxes:
            corner, height, width = deboxing(bbox)
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[0, 1, 0], facecolor='none'
            )
            ax.add_patch(rect)
    if pred_boxes is not None:
        for bbox in pred_boxes:
            corner, height, width = deboxing(bbox)
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[1, 0, 0], facecolor='none'
            )
            ax.add_patch(rect)

    plt.show()


def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, checkpoint_path)

    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(checkpoint_path, best_model_path)


def collate_fn(batch):
    return tuple(zip(*batch))


def g_to_rgb(image):
    a = image.to_numpy()
    b = np.repeat(a[:, :, np.newaxis], 3, axis=2)
    rer_b = np.transpose(b, axes=[2, 0, 1])
    return rer_b
