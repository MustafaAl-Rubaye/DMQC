import cv2
import glob
import matplotlib.pyplot as plt
import os

img_path = (
    "/home/mustafa/Documents/Mamography/data_/images/Mass-Training_P_00004_LEFT_MLO.png"
)

img_save_dir = "/home/mustafa/Documents/Mamography/data/500_img_resized/"
mask_save_dir = "/home/mustafa/Documents/Mamography/data/small_masks/"

os.makedirs(img_save_dir, exist_ok=True)
os.makedirs(mask_save_dir, exist_ok=True)

img_files = glob.glob("/home/mustafa/Documents/Mamography/data/the_500_images/*.png")
masks_file = glob.glob("/home/mustafa/Documents/Mamography/data/masks/*.png")

for img_path in img_files:
    im = cv2.imread(img_path)
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

    cv2.imwrite(img_save_dir + img_path.split("/", -1)[-1], im)
    # cv2.imwrite(mask_save_dir + file.split('/', -1)[-1][:-5], masks)

    # cv2.imshow("image", new_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.imshow(im)
    plt.show()

# plt.imshow(new_im)
# plt.show()
