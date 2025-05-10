# this code take the dots from the json files and make polygons
# AND THEN save the images

import math
import pandas as pd
import os
import glob
import numpy as np
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


save_dir = "/home/mustafa/Documents/Mamography/data/"
os.makedirs(save_dir, exist_ok=True)
files = glob.glob("/home/mustafa/Documents/Mamography/raw_segmentation_f/*.json")
result = []
for file in files:
    with open(file) as json_file:
        # ori_data = json.load(json_file)
        result.append(json.load(json_file))

with open("merged_file.json", "w") as outfile:
    json.dump(result, outfile)


file = "/home/mustafa/Documents/Mamography/pre_processing/merged_file.json"


def testing(file):
    with open(file) as json_file:

        ori_data = json.load(json_file)

        print(ori_data.shape())
