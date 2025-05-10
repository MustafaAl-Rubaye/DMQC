import argparse
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import solt
import torch
from sklearn.model_selection import GroupKFold, train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from dmqc.data.dataset import MammographySegmentationDataset
from dmqc.pipeline.helper import (
    path_to_df_from_config,
    dice_list,
    find_the_best_threshold,
)
import segmentation_models_pytorch as smp
import yaml
from collections import OrderedDict


def fold_split_data(confid):
    df = path_to_df_from_config(config)

    dataframe_train_val, dataframe_test = train_test_split(
        df, test_size=0.2, random_state=config["seed"],
    )

    kfold = GroupKFold(n_splits=config["n_folds"])

    for train, val in kfold.split(
        dataframe_train_val, groups=dataframe_train_val["id"]
    ):

        train_df = dataframe_train_val.iloc[train]
        val_df = dataframe_train_val.iloc[val]

        return train_df, val_df


def best_threshold(config):

    for set_ in meta_probs:
        wb = (
            set_[0]
            .sigmoid()
            .ge(config["best_threshold"]["wb"])
            .to("cpu")
            .float()
            .numpy()
            .astype(np.uint8)
        )
        b = (
            set_[1]
            .sigmoid()
            .ge(config["best_threshold"]["b"])
            .to("cpu")
            .float()
            .numpy()
            .astype(np.uint8)
        )
        m = (
            set_[2]
            .sigmoid()
            .ge(config["best_threshold"]["m"])
            .to("cpu")
            .float()
            .numpy()
            .astype(np.uint8)
        )
        sf = (
            set_[3]
            .sigmoid()
            .ge(config["best_threshold"]["sf"])
            .to("cpu")
            .float()
            .numpy()
            .astype(np.uint8)
        )

        n = (
            set_[4]
            .sigmoid()
            .ge(config["best_threshold"]["n"])
            .to("cpu")
            .float()
            .numpy()
            .astype(np.uint8)
        )
        thresholded_probs.append([wb, b, m, sf, n])

    # best_threshold_list = find_the_best_threshold(
    #     meta_probs, labels, batch
    # )

    return thresholded_probs


def init_loader(metadata_df, config):
    transforms = solt.Stream()

    dataset = MammographySegmentationDataset(
        metadata_df, transforms, config["data"]["json_path"]
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=config["bs"],
        num_workers=config["num_workers"],
        sampler=sampler,
    )

    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/oof/"),
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=Path(
            "/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/6_mod_300_epo/best_mode/"
        ),
    )
    args = parser.parse_args()
    # model_wise
    save_list = []
    # for run_i in os.listdir(args.snapshots):
    #     print(1)
    for model_i in os.listdir(args.snapshots):
        snapshot_path = Path(os.path.join(args.snapshots, Path(model_i)))

        snps_paths = list(snapshot_path.glob("*/"))

        with open(os.path.join(snps_paths[0], "config.yaml"), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        classes_all = []
        targets_all = []
        probs_all = []
        idxs = []

        check_ponts = snps_paths[2].glob("**/*.ckpt")
        fold_wise_dice_list = []
        fold_wise_standard_error_list = []

        fold_wise_threshold_list = []
        for checkpoint_path in check_ponts:
            checkpoint = torch.load(checkpoint_path)

            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])
            random.seed(config["seed"])

            _, val_ds = fold_split_data(config)

            sampler = SequentialSampler(val_ds)
            loader = init_loader(val_ds, config)

            models = {
                "FPN": smp.FPN,
                "UNet": smp.Unet,
            }

            model = models[config["backbone"]](
                config["model"],
                classes=config["data"]["n_classes"],
                activation="identity",
            )

            optimizer = torch.optim.Adam(
                model.parameters(), lr=config["lr"], weight_decay=config["wd"]
            )

            new_state_dict = OrderedDict()

            for k, v in checkpoint["state_dict"].items():
                name = k[6:]  # remove `model.`
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)

            optimizer.load_state_dict(checkpoint["optimizer_states"][0])

            model = torch.nn.DataParallel(model).to("cuda")
            model.eval()

            with torch.no_grad():
                batch_wise_list = []
                batch_wise_standard_error = []

                batch_wise_list_for_threshold = []

                for batch in tqdm(loader, total=len(loader)):
                    data = batch["img"]
                    targets = batch["masks"]  # .numpy().tolist()

                    labels = (
                        np.squeeze(targets, axis=2).to("cpu").numpy().astype(np.uint8)
                    )

                    meta_probs = model(data)

                    # probs = (
                    #     meta_probs.sigmoid()
                    #     .ge(0.5)
                    #     .to("cpu")
                    #     .float()
                    #     .numpy()
                    #     .astype(np.uint8)
                    # )

                    thresholded_probs = []
                    thresholded_list = best_threshold(config)

                    # batch_wise_list_for_threshold.append(best_threshold_list)

                    batch_dice, batch_standard_error = dice_list(
                        labels=labels,
                        prediction=thresholded_list,
                        classes=batch["classes_"],
                    )

                    batch_wise_list.append(batch_dice)
                    batch_wise_standard_error.append(batch_standard_error)

                # fold_threshold = np.array(batch_wise_list_for_threshold).mean(axis=0)
                # fold_wise_threshold_list.append(fold_threshold)

                fold_dice = np.array(batch_wise_list).mean(axis=0)
                fold_standard_error = np.array(batch_wise_standard_error).mean(axis=0)

                fold_wise_dice_list.append(fold_dice)
                fold_wise_standard_error_list.append(fold_standard_error)

        # model_wise_threshold = np.array(fold_wise_threshold_list).mean(axis=0)

        model_wise_dice = np.array(fold_wise_dice_list).mean(axis=0)
        model_wise_standard_error = np.array(fold_wise_standard_error_list).mean(axis=0)

        # print_the_best_threshold = (
        #     f"Architecture: {config['backbone']} |"
        #     f"Model: {config['model']} |"
        #     f"Best threshold for Whole Breast :{round(model_wise_threshold[0], 2)} | "
        #     f"Best threshold for Breast : {round(model_wise_threshold[1], 2)} | "
        #     f"Best threshold for Muscle: {round(model_wise_threshold[2], 2)} | "
        #     f"Best threshold for Skin Folding: {round(model_wise_threshold[3], 2)} | "
        #     f"Best threshold for Nipple: {round(model_wise_threshold[4], 2)}"
        # )
        #
        # print(print_the_best_threshold)

        print_the_dice = (
            f"Architecture: {config['backbone']} |"
            f"Model: {config['model']} |"
            f"Whole Breast: {round(model_wise_dice[0], 3)} ± {round(model_wise_standard_error[0], 3)}| "
            f"Breast: {round(model_wise_dice[1], 3)} ± {round(model_wise_standard_error[1], 3)}| "
            f"Muscle: {round(model_wise_dice[2], 3)} ± {round(model_wise_standard_error[2], 3)}| "
            f"Skin Folding: {round(model_wise_dice[3], 3)} ± {round(model_wise_standard_error[3], 3)}| "
            f"Nipple: {round(model_wise_dice[4], 3)} ± {round(model_wise_standard_error[4], 3)}"
        )

        print(print_the_dice)

        # to_print = (
        #     f"[{round(model_wise_standard_error[0], 6)},"
        #     f" {round(model_wise_standard_error[1], 6)},"
        #     f" {round(model_wise_standard_error[2], 6)},"
        #     f" {round(model_wise_standard_error[3], 6)},"
        #     f" {round(model_wise_standard_error[4], 6)}]"
        # )
        #
        # print(to_print)

    #     save_list.append({
    #         "Architecture": [config['backbone']],
    #         "Model": {config['model']},
    #         "Best threshold for Whole Breast": {round(model_wise_threshold[0], 2)},
    #         "Best threshold for Breast": {round(model_wise_threshold[1], 2)},
    #         "Best threshold for Muscle": {round(model_wise_threshold[2], 2)},
    #         "Best threshold for Skin Folding": {round(model_wise_threshold[3], 2)},
    #         "Best threshold for Nipple": {round(model_wise_threshold[4], 2)},
    #     })
    #
    #
    #
    # df = pd.DataFrame(data=save_list)
    # df.to_csv(os.path.join(args.save_dir, "threshold_metadata.csv"), index=None)
