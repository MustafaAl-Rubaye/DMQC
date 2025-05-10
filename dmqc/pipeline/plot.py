import argparse
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from dmqc.pipeline.helper import zip
from pathlib import Path

SEED = 42
np.random.seed(SEED)


def plot_sample(X, y, preds, binary_preds, ix=None):

    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap="seismic")
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors="k", levels=[0.5])
    ax[0].set_title("Seismic")

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title("Salt")

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors="k", levels=[0.5])
    ax[2].set_title("Salt Predicted")

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors="k", levels=[0.5])
    ax[3].set_title("Salt Predicted binary")


def plot(X_train, y_train, results):
    ix = random.randint(0, len(X_train))
    has_mask = y_train[ix].max() > 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))
    ax1.imshow(X_train[ix, ..., 0], cmap="seismic", interpolation="bilinear")
    if has_mask:
        ax1.contour(y_train[ix].squeeze(), colors="k", linewidths=5, levels=[0.5])
    ax1.set_title("Image")

    ax2.imshow(y_train[ix].squeeze(), cmap="gray", interpolation="bilinear")
    ax2.set_title("Mask")

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(
        np.argmin(results.history["val_loss"]),
        np.min(results.history["val_loss"]),
        marker="x",
        color="r",
        label="best model",
    )
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.show()


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


def denormalize(x):
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def plot_roc(gt, preds):
    fpr1, tpr1, _ = roc_curve(gt, preds)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr1, tpr1, lw=2, c="b")
    plt.plot([0, 1], [0, 1], "--", c="black")
    plt.grid()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def loss_plot(train_losses, val_losses):
    plt.figure(figsize=(10, 10))
    plt.plot(train_losses, "r", label="train loss")
    plt.plot(val_losses, "b", label="val loss")
    plt.legend()
    plt.show()


def dice_class_plot(val_dice, n_classes):
    val_dice_np = np.asarray(val_dice)
    plt.figure(figsize=(10, 10))
    # for i in range(n_classes):
    plt.plot(val_dice_np)  # , label=label_list[i])
    # plt.plot(val_dice_np, 'b', label='val loss')
    plt.suptitle("Dice for different classes")
    plt.legend()
    plt.show()


def plot_prediction(input_cpu, prediction_cpu, mask_names):
    data = zip(next(iter(input_cpu)), prediction_cpu, mask_names)
    data_list = list(data)
    # data_i = data_list[0]
    # for data_i in data_list:
    data_i = data_list[0]
    data_i_img = np.squeeze(data_i[0])
    data_i_mask = np.squeeze(data_i[1])
    data_i_name = data_i[2].split("/")[-1][:-4]

    fig, a = plt.subplots(1, 2)

    # data_i_img_sqeee = np.squeeze(data_i_img)
    a[0].imshow(data_i_img, cmap="gray")
    a[0].set_title(data_i_name + " Image")

    a[1].imshow(data_i_mask, cmap="gray")
    a[1].set_title(data_i_name + " mask")

    plt.show()


def show_prediction_Contours(input_cpu, prediction_cpu, mask_names):

    data_i_img = np.squeeze(input_cpu[0])
    data_i_mask = np.squeeze(prediction_cpu[0])
    data_i_name = mask_names[0].split("/")[-1][:-4]

    img_1 = data_i_img.copy()
    img_2 = img_1.copy()
    img_3 = img_2.copy()
    img_4 = data_i_img.copy()
    img_5 = data_i_img.copy()
    img = data_i_img.copy()

    masks = []
    for i in range(len(data_i_mask)):
        contours, _ = cv2.findContours(
            data_i_mask[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        masks.append(contours)

    img_with_breast_contors = cv2.drawContours(img_2, masks[1], -1, (255, 0, 0), 5)
    img_with_whole_breast_contors = cv2.drawContours(
        img_1, masks[0], -1, (255, 255, 0), 5
    )
    img_with_muscle_contors = cv2.drawContours(img_3, masks[2], -1, (0, 255, 0), 5)
    img_with_nipple_contors = cv2.drawContours(img_4, masks[4], -1, (0, 255, 255), 5)
    img_with_skin_folding_contors = cv2.drawContours(
        img_5, masks[3], -1, (127, 0, 255), 5
    )

    plt.subplot(231), plt.imshow(img)
    plt.title("Ori " + data_i_name), plt.xticks([]), plt.yticks([])

    plt.subplot(232), plt.imshow(img_with_whole_breast_contors)
    plt.title("Whole Breast"), plt.xticks([]), plt.yticks([])

    plt.subplot(233), plt.imshow(img_with_breast_contors)
    plt.title("Breast"), plt.xticks([]), plt.yticks([])

    plt.subplot(234), plt.imshow(img_with_muscle_contors)
    plt.title("Muscle"), plt.xticks([]), plt.yticks([])

    plt.subplot(235), plt.imshow(img_with_nipple_contors)
    plt.title("nipple"), plt.xticks([]), plt.yticks([])

    plt.subplot(236), plt.imshow(img_with_skin_folding_contors)
    plt.title(" Skin Folding"), plt.xticks([]), plt.yticks([])

    plt.show()


def plot_multichannel_prediction(input_cpu, prediction_cpu, mask_names):
    data = zip(next(iter(input_cpu)), prediction_cpu, mask_names)
    data_list = list(data)
    data_i = data_list[0]
    data_i_img = np.squeeze(data_i[0])
    data_i_mask = np.squeeze(data_i[1])

    plt.subplot(231), plt.imshow(data_i_img, cmap="gray")
    plt.title("Ori"), plt.xticks([]), plt.yticks([])

    plt.subplot(232), plt.imshow(data_i_mask[0], cmap="gray")
    plt.title("Whole Breast"), plt.xticks([]), plt.yticks([])

    plt.subplot(233), plt.imshow(data_i_mask[1], cmap="gray")
    plt.title("Breast"), plt.xticks([]), plt.yticks([])

    plt.subplot(234), plt.imshow(data_i_mask[2], cmap="gray")
    plt.title("Muscle"), plt.xticks([]), plt.yticks([])

    plt.subplot(235), plt.imshow(data_i_mask[4], cmap="gray")
    plt.title("nipple"), plt.xticks([]), plt.yticks([])

    plt.subplot(236), plt.imshow(data_i_mask[3], cmap="gray")
    plt.title(" Skin Folding"), plt.xticks([]), plt.yticks([])

    plt.show()


def dice_histogram(val_dice, n_classes):
    val_dice_np = np.asarray(val_dice)
    plt.figure(figsize=(10, 10))

    plt.hist(val_dice_np)
    plt.suptitle("Dice for different classes")
    plt.legend()
    plt.show()


def plot_the_mean(files):
    data_list = []
    for file_i in os.listdir(files):
        file_i_path = Path(os.path.join(files, Path(file_i)))
        # with open(file_i_path) as csv_file:
        data = pd.read_csv(file_i_path)

        data_list.append(data)

    mean_data = pd.concat(data_list).groupby(level=0).mean()

    steps = np.array(mean_data.Step)
    values = np.array(mean_data.Value)

    plt.figure(figsize=(20, 10))
    # plt.plot(steps, values)
    # plt.plot([1, 2, 3, 4], 'ro')
    plt.plot(steps, values)
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')

    # plt.ylabel('some numbers')
    plt.show()

    return mean_data


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--dir",
#         type=Path,
#         default=Path(
#             "/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/6_mod_300_epo/oof/plot_data/"
#         ),
#     )
#     parser.add_argument(
#         "--snapshots",
#         type=Path,
#         default=Path(
#             "/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/6_mod_300_epo/1/"
#         ),
#     )
#     args = parser.parse_args()
#
#     mean_data = plot_the_mean(args.dir)
