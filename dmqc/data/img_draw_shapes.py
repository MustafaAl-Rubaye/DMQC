# this code take the dots from the json files and make polygons
# AND THEN save the images (masks)

import os
import glob
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from dmqc import the_classes
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_dir", default="/home/mustafa/Documents/Mamography/data/test_the_500_masks/"
)
parser.add_argument(
    "--files", default="/home/mustafa/Documents/Mamography/raw_segmentation_f/*.json"
)
parser.add_argument("--n_samples", type=int, default=5000)
args = parser.parse_args()


save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
files = glob.glob(args.files)

for file in files:
    with open(file) as json_file:

        ori_data = json.load(json_file)
        images = len(ori_data["images"])

        for i in tqdm(range(len(ori_data["images"]))):
            H = ori_data["images"][i]["height"]
            W = ori_data["images"][i]["width"]
            image_name = ori_data["images"][i]["file_name"]
            img = np.zeros((H, W), dtype=np.uint8)

            # test = len(ori_data["annotations"])

            for ids in range(len(ori_data["annotations"])):
                image_id = ori_data["annotations"][ids]["image_id"]
                list_points = ori_data["annotations"][ids]["segmentation"]

                flatten_list_pints = sum(list_points, [])

                x = 0
                new_list = []
                while x < len(flatten_list_pints):
                    new_list.append(flatten_list_pints[x : x + 2])
                    x += 2
                class_title = ori_data["annotations"][ids]["category_id"]

                if class_title == 1:
                    class_title = "Breast"

                if class_title == 2:
                    class_title = "Muscle"

                if class_title == 3:
                    class_title = "Nipple"

                if class_title == 4:
                    class_title = "Skin_Folding"

                if class_title == 5:
                    class_title = "Whole_Breast"

                number_of_i = i
                number_of_id = ids

                if image_id == i:

                    np_points = np.array(new_list)[None, ...].astype(np.int32)
                    tmp = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(tmp, np_points, 1)
                    img[tmp > 0] = the_classes[class_title]

                    # cv2.imwrite(save_dir + image_name, img)
                    plt.imshow(img, cmap=plt.cm.Greys, vmin=0, vmax=5)

                    plt.show()
