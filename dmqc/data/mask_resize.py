import cv2
import glob
import matplotlib.pyplot as plt
import os

mask_save_dir = "/home/mustafa/Documents/Mamography/data/500_mask_resized/"

os.makedirs(mask_save_dir, exist_ok=True)

masks_file = glob.glob("/home/mustafa/Documents/Mamography/data/the_500_masks/*.png")

for mask_path in masks_file:
    im = cv2.imread(mask_path)
    old_dim = im.shape

    old_H = old_dim[0]
    old_W = old_dim[1]

    w = 512
    h = 256

    if old_W < old_H // 2:
        delta = old_H // 2 - old_W
        top, bottom = 0, 0
        right, left = delta // 2, delta // 2

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(
            src=im,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=color,
        )

    else:
        delta_h = 2 * old_W - old_H
        top, bottom = delta_h // 2, delta_h // 2
        left, right = 0, 0

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(
            src=im,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=color,
        )

    im = cv2.resize(new_im, (h, w))

    cv2.imwrite(mask_save_dir + mask_path.split("/", -1)[-1], im)

    plt.imshow(im)
    plt.show()
