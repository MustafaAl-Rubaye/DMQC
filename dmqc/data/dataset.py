import cv2
import torch
from torch.utils import data
import numpy as np
from dmqc.data.masks_generator import get_masks_from_json
import json

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class MammographySegmentationDataset(data.Dataset):
    def __init__(self, metadata, transforms, json_file):
        self.metadata = metadata
        self.transform = transforms
        self.json_file = json_file

        with open(self.json_file) as json_file:
            ori_data = json.load(json_file)

        self.ori_data = ori_data

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        meta = self.metadata.iloc[idx]
        img = cv2.imread(str(meta.images))
        image_ID = meta.images.name

        masks_list, classes = get_masks_from_json(self.ori_data, image_ID)

        res_dc = self.transform(
            {"image": img, "masks": masks_list},
            return_torch=True,
            mean=self.mean,
            std=self.std,
        )
        img_res = res_dc["image"]
        masks_res = res_dc["masks"]
        masks = torch.stack(masks_res)

        return {
            "img": img_res,
            "masks": masks,
            "img_name": str(meta.images),
            "mask_name": str(masks),
            "classes_": classes,
        }
