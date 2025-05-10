import pydicom as dicom
import matplotlib.pyplot as plt
import os
import cv2
import pylab
import numpy as np

import glob

folder_path = "give the folder path here"
files = glob.glob("folder_path/*.dcm")

save_path = "add save path path"  # the images will be saved as png:s


def dicom_to_png(files):

    for image in files:
        # if image.endswith(".dcm"):
        ds = dicom.dcmread(os.path.join(folder_path, image))
        pixel_array_numpy = ds.pixel_array * 6

        image = image.replace(".dcm", ".png")
        cv2.imwrite(os.path.join(save_path, image), pixel_array_numpy)
        plt.imshow(pixel_array_numpy, cmap=pylab.cm.bone)

        plt.show()


dicom_to_png(files)
