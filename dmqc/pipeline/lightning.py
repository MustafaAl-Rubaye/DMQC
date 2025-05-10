import torch
import numpy as np
import torchvision
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import solt
import solt.transforms as slt
from dmqc.data.dataset import MammographySegmentationDataset
import pytorch_lightning as pl
from dmqc.pipeline.helper import dice_list
from dmqc.data.calculate_class_weights import calculate_weights
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from dmqc.pipeline.helper import zip
from pandas import DataFrame
import json


def my_transforms():
    train_stream = solt.Stream(
        [
            slt.Pad((270, 520)),
            slt.Flip(axis=1, p=0.5),
            slt.Crop((256, 512), crop_mode="r"),
            # solt.SelectiveStream([
            #     slt.GammaCorrection(gamma_range=0.5, p=1),
            #     slt.Noise(gain_range=0.1, p=1),
            #     slt.Blur()
            #     ], n=3)
        ]
    )

    test_stream = solt.Stream()

    return {"train": train_stream, "eval": test_stream}


class Mylightningclass(pl.LightningModule):
    def __init__(self, val_df, val_stream, train_df, train_stream, cfg, test_df=None):
        super().__init__()
        self.lr = cfg.lr
        self.val_df = val_df
        self.train_df = train_df
        self.n_classes = cfg.data.n_classes
        self.weight_decay = cfg.wd
        self.json_path = cfg.data.json_path
        self.threshold = cfg.threshold
        self.batch_size = cfg.bs
        self.test_df = test_df
        self.num_workers = cfg.num_workers
        self.milestones = list(cfg.milestones)

        models = {
            "FPN": smp.FPN,
            "UNet": smp.Unet,
        }

        self.model = models[cfg.backbone](
            cfg.model, classes=cfg.data.n_classes, activation="identity"
        )
        self.n_epochs = cfg.n_epoches
        self.train_stream = train_stream
        self.val_stream = val_stream
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        # Setting up the class weights
        self.class_weights = torch.tensor([1.0, 2.0, 10.0, 100.0, 350.0])
        # First initial guess of weights was: [1., 1., 1., 50., 100.]
        # Normalizing the weights to have their sum to be one
        self.class_weights.div_(self.class_weights.sum()).unsqueeze(0)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def forward(self, x):
        return self.model(x)

    def compute_loss_with_class_weights(self, prediction, labels):
        loss = self.criterion(prediction, labels)
        if self.class_weights.device != loss.device:
            self.class_weights = self.class_weights.to(loss.device)
        loss = loss.mean(dim=[2, 3]).mul(self.class_weights)

        return loss

    def compute_loss(self, prediction, labels):
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(prediction, labels)

        return loss

    def training_step(self, batch, batch_idx):
        labels = batch["masks"]
        labels = np.squeeze(labels, axis=2)
        # weight_list = []
        # classes_weight = calculate_weights(labels)
        #
        # weight_list.append(classes_weight)

        inputs = batch["img"]

        prediction = self.model(inputs)

        loss = self.compute_loss_with_class_weights(prediction, labels)

        return {"loss": loss}

    def train_dataloader(self):
        train_ds = MammographySegmentationDataset(
            self.train_df, self.train_stream, self.json_path
        )

        train_loader = DataLoader(
            train_ds, self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        return train_loader

    def validation_step(self, batch, batch_idx):

        labels = batch["masks"]
        labels = np.squeeze(labels, axis=2)
        inputs = batch["img"]
        mask_names = batch["mask_name"]

        prediction = self.model(inputs)

        loss = self.compute_loss(prediction, labels)

        return {
            "loss": loss,
            "prediction": prediction,
            "inputs": inputs,
            "batch": batch,
            "labels": labels,
            "names": mask_names,
            # "batch_idx": batch_idx,
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.cat([x["loss"] for x in outputs]).mean()

        log = {"avg_val_loss": val_loss}

        return {"log": log}

    def val_dataloader(self):
        val_ds = MammographySegmentationDataset(
            self.val_df, self.val_stream, self.json_path
        )

        val_loader = DataLoader(val_ds, self.batch_size, num_workers=self.num_workers)
        return val_loader

    def test_step(self, batch, batch_idx):

        labels = batch["masks"]
        labels = np.squeeze(labels, axis=2)
        inputs = batch["img"]
        images_paths = [i.split("/")[-1][:-4] for i in batch["img_name"]]

        print(images_paths)

        # results_ = list(map(int, images_paths))
        # images_paths = batch["img_name"]
        # data_frame = DataFrame(images_paths, columns=['path'])
        # image_tensor = torch.as_tensor(results_)

        # self.name_list = images_paths

        prediction = self.model(inputs)

        loss = self.compute_loss(prediction, labels)

        return {
            "loss": loss,
            "prediction": prediction,
            "inputs": inputs,
            "batch": batch,
            "labels": labels,
            "names": images_paths,
        }

    def test_epoch_end(self, outputs):
        batch_wise_list = []
        for one_bat, batch in enumerate(outputs):
            test_loss = torch.cat([x["loss"] for x in outputs]).mean()
            preds = batch["prediction"].sigmoid().ge(self.threshold).to("cpu")
            prediction = preds.float().numpy().astype(np.uint8)

            # inputs = batch["inputs"]
            # data_mean = torch.tensor(self.mean).view(3, 1, 1).to(inputs.device)
            # data_std = torch.tensor(self.std).view(3, 1, 1).to(inputs.device)
            # inputs_ = inputs.mul(data_std).add(data_mean).mul(255)

            labels = batch["labels"].to("cpu").numpy().astype(np.uint8)

            # self.save_image(prediction, one_bat)
            self.save_to_json(prediction, one_bat)

            batch_dice = dice_list(
                labels=labels, prediction=prediction, classes=batch["batch"]["classes_"]
            )

            batch_wise_list.append(batch_dice)

        epoch_dice, error = np.array(batch_wise_list).mean(axis=0)

        dice = (
            f"Whole Breast: {epoch_dice[0]} ± {error[0]} | "
            f"Breast: {epoch_dice[1]} ± {error[0]} | "
            f"Muscle: {epoch_dice[2]} ± {error[0]} | "
            f"Skin Folding: {epoch_dice[3]} ± {error[0]} | "
            f"Nipple: {epoch_dice[4]} ± {error[0]}"
        )

        print(dice)

        # image = torchvision.utils.make_grid(inputs_[0])
        # self.logger.experiment.add_image("image", image, 3)
        #
        # wb_img = torchvision.utils.make_grid(preds[0][0])
        # self.logger.experiment.add_image("Whole Breast", wb_img, 3)
        #
        # breast_img = torchvision.utils.make_grid(preds[0][1])
        # self.logger.experiment.add_image("Breast", breast_img, 3)
        #
        # m_imag = torchvision.utils.make_grid(preds[0][2])
        # self.logger.experiment.add_image("Muscle", m_imag, 3)
        #
        # s_f_img = torchvision.utils.make_grid(preds[0][3])
        # self.logger.experiment.add_image("Skin Folding", s_f_img, 3)
        #
        # n_img = torchvision.utils.make_grid(preds[0][4])
        # self.logger.experiment.add_image("Nipple", n_img, 3)
        #
        # self.logger.experiment.add_scalars(
        #     "Dice",
        #     {
        #         "Whole Breast": epoch_dice[0],
        #         "Breast": epoch_dice[1],
        #         "Muscle": epoch_dice[2],
        #         "Skin Folding": epoch_dice[3],
        #         "Nipple": epoch_dice[4],
        #     },
        #     self.current_epoch,
        # )

        log = {"avg_test_loss": test_loss}

        return {"test_log": log}

    def test_dataloader(self):
        test_ds = MammographySegmentationDataset(
            self.test_df, self.val_stream, self.json_path
        )

        test_loader = DataLoader(test_ds, self.batch_size, num_workers=self.num_workers)
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return (
            [optimizer],
            [
                {
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                        optimizer=optimizer, milestones=self.milestones, gamma=0.1
                    ),
                    "interval": "epoch",
                    "monitor": "avg_val_loss",
                }
            ],
        )

        # return optimizer

    def save_image(self, prediction, batch_num):

        names = [
            [
                "4285",
                "1799",
                "4498",
                "2934",
                "2126",
                "4405",
                "5624",
                "2441",
                "1757",
                "4035",
                "1041",
                "3433",
                "4696",
                "1898",
                "5465",
                "5614",
                "942",
                "1237",
                "4827",
                "579",
                "87",
                "893",
                "3844",
                "3919",
                "1819",
                "6437",
                "655",
                "5740",
                "2099",
                "5022",
                "4599",
                "3821",
            ],
            [
                "657",
                "3215",
                "1027",
                "5019",
                "4333",
                "1777",
                "3678",
                "1692",
                "5125",
                "2023",
                "844",
                "3330",
                "1277",
                "1827",
                "1005",
                "1077",
                "4743",
                "1161",
                "1794",
                "3290",
                "2528",
                "5869",
                "3426",
                "1611",
                "2727",
                "1137",
                "598",
                "1854",
                "940",
                "5857",
                "4877",
                "833",
            ],
            [
                "1870",
                "1314",
                "6251",
                "1415",
                "4267",
                "5627",
                "5223",
                "366",
                "4376",
                "4514",
                "4837",
                "4705",
                "4129",
                "1758",
                "861",
                "4125",
                "2920",
                "1887",
                "4683",
                "3212",
                "4863",
                "2001",
                "3282",
                "5904",
                "6486",
                "570",
                "5601",
                "671",
                "6104",
                "4882",
                "6041",
                "967",
            ],
            ["1941", "1814", "6111", "1131", "4097"],
        ]

        # name_testing = self.name_list

        for i_pred in range(len(prediction)):

            for i in range(prediction[i_pred][0].shape[0]):
                for j in range(prediction[i_pred][0].shape[1]):
                    if prediction[i_pred][0][i, j] == 1:
                        prediction[i_pred][0][i, j] = 40

                    if prediction[i_pred][1][i, j] == 1:
                        prediction[i_pred][1][i, j] = 80

                    if prediction[i_pred][2][i, j] == 1:
                        prediction[i_pred][2][i, j] = 120

                    if prediction[i_pred][3][i, j] == 1:
                        prediction[i_pred][3][i, j] = 160

                    if prediction[i_pred][4][i, j] == 1:
                        prediction[i_pred][4][i, j] = 200

            img = (
                prediction[i_pred][0]
                + prediction[i_pred][1]
                + prediction[i_pred][2]
                + prediction[i_pred][3]
                + prediction[i_pred][4]
            )
            save_dir = "/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/6_mod_300_epo/masks/"
            cv2.imwrite(os.path.join(save_dir, f"{names[batch_num][i_pred]}.png"), img)
            # plt.imshow(testing, cmap="gray")
            #
            # plt.show()

    def save_to_json(self, prediction, batch_num):

        data = {}
        # data['whole breast'] = []
        # data['breast'] = []
        # data['muscle'] = []
        # data['skinfolds'] = []
        # data['nipple'] = []

        names = [
            [
                "4285",
                "1799",
                "4498",
                "2934",
                "2126",
                "4405",
                "5624",
                "2441",
                "1757",
                "4035",
                "1041",
                "3433",
                "4696",
                "1898",
                "5465",
                "5614",
                "942",
                "1237",
                "4827",
                "579",
                "87",
                "893",
                "3844",
                "3919",
                "1819",
                "6437",
                "655",
                "5740",
                "2099",
                "5022",
                "4599",
                "3821",
            ],
            [
                "657",
                "3215",
                "1027",
                "5019",
                "4333",
                "1777",
                "3678",
                "1692",
                "5125",
                "2023",
                "844",
                "3330",
                "1277",
                "1827",
                "1005",
                "1077",
                "4743",
                "1161",
                "1794",
                "3290",
                "2528",
                "5869",
                "3426",
                "1611",
                "2727",
                "1137",
                "598",
                "1854",
                "940",
                "5857",
                "4877",
                "833",
            ],
            [
                "1870",
                "1314",
                "6251",
                "1415",
                "4267",
                "5627",
                "5223",
                "366",
                "4376",
                "4514",
                "4837",
                "4705",
                "4129",
                "1758",
                "861",
                "4125",
                "2920",
                "1887",
                "4683",
                "3212",
                "4863",
                "2001",
                "3282",
                "5904",
                "6486",
                "570",
                "5601",
                "671",
                "6104",
                "4882",
                "6041",
                "967",
            ],
            ["1941", "1814", "6111", "1131", "4097"],
        ]

        for i_pred in range(len(prediction)):

            data["whole breast"] = prediction[i_pred][0].tolist()
            data["breast"] = prediction[i_pred][1].tolist()
            data["muscle"] = prediction[i_pred][2].tolist()
            data["skinfolds"] = prediction[i_pred][3].tolist()
            data["nipple"] = prediction[i_pred][4].tolist()

            save_dir = "/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/6_mod_300_epo/masks_json/"

            with open(
                os.path.join(save_dir, f"{names[batch_num][i_pred]}.json"), "w"
            ) as outfile:
                json.dump(data, outfile)
            # cv2.imwrite(os.path.join(save_dir, f"{names[batch_num][i_pred]}.png"), img)
            # plt.imshow(testing, cmap="gray")
            #
            # plt.show()
