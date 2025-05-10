import pathlib
import pandas as pd
import argparse
import numpy as np
import math
import os
from dmqc.pipeline.metrics import (
    calculate_dice,
    calculate_confusion_matrix_from_arrays,
)


def get_meta():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/mustafa/Documents/Mammo_project/data/metadata_processed.csv",
    )
    args = parser.parse_args()

    file_path = args.root

    df = pd.read_csv(file_path)

    return df


def zip(*iterables):
    # zip('ABCD', 'xy') --> Ax By
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)


def path_to_df(cfg):
    data_dir = pathlib.Path(cfg.data.dir)

    img_paths = data_dir.glob(cfg.data.img_path)
    img_sorted = sorted([x for x in img_paths])

    """
    I used the following code to to make to make csv file of the patient ids
    because making the dataframe eveytime will take too much time
    patient_meta = get_meta()
    id_info = []
    for x in img_sorted:
        for w in patient_meta.iterrows():
            name = int(x.name[:-4])
            if name == w[1].ID:

                id_info.append(
                    {
                        "patient_id": w[1].patient_id,
                    }
                )

                df = pd.DataFrame(data=id_info)
                df.to_csv(os.path.join(args.save_dir, "patient_id.csv"), index=None)
    """

    patient_id = pd.read_csv(cfg.data.patient_ids)
    datafram = {"images": img_sorted, "id": patient_id.values.tolist()}
    df = pd.DataFrame(data=datafram)

    return df


def dice_list(labels, prediction, classes):
    wb_dice = []
    b_dice = []
    m_dice = []
    sf_dice = []
    n_dice = []
    # for one_ in range(len(self.val_df)/self.batch_size):
    for sample_i in range(labels.shape[0]):
        pred_cpu = prediction[sample_i]
        l_cpu = labels[sample_i]
        cm_WB = calculate_confusion_matrix_from_arrays(pred_cpu[0], l_cpu[0], 2)
        cm_B = calculate_confusion_matrix_from_arrays(pred_cpu[1], l_cpu[1], 2)
        cm_M = calculate_confusion_matrix_from_arrays(pred_cpu[2], l_cpu[2], 2)
        cm_SF = calculate_confusion_matrix_from_arrays(pred_cpu[3], l_cpu[3], 2)
        cm_N = calculate_confusion_matrix_from_arrays(pred_cpu[4], l_cpu[4], 2)

        dice_WB = calculate_dice(cm_WB)
        dice_B = calculate_dice(cm_B)
        dice_M = calculate_dice(cm_M)
        dice_SF = calculate_dice(cm_SF)
        dice_N = calculate_dice(cm_N)

        wb_dice.append(dice_WB[1])
        b_dice.append(dice_B[1])
        m_dice.append(dice_M[1])
        sf_dice.append(dice_SF[1])
        n_dice.append(dice_N[1])

        wb_no_zero_dice = []
        for dice_1 in wb_dice:
            if dice_1 == 0:
                if int(classes["whole breast"][sample_i]) == 1:
                    wb_no_zero_dice.append(dice_1)
                else:
                    pass
            else:
                wb_no_zero_dice.append(dice_1)

        b_no_zero_dice = []
        for dice_2 in b_dice:
            if dice_2 == 0:
                if int(classes["breast"][sample_i]) == 1:
                    b_no_zero_dice.append(dice_2)
                else:
                    pass
            else:
                b_no_zero_dice.append(dice_2)

        m_no_zero_dice = []
        for dice_3 in m_dice:
            if dice_3 == 0:
                if int(classes["muscle"][sample_i]) == 1:
                    m_no_zero_dice.append(dice_3)
                else:
                    pass
            else:
                m_no_zero_dice.append(dice_3)

        sf_no_zero_dice = []
        for dice_4 in sf_dice:
            if dice_4 == 0:
                if int(classes["skin folding"][sample_i]) == 1:
                    sf_no_zero_dice.append(dice_4)
                else:
                    pass
            else:
                sf_no_zero_dice.append(dice_4)

        n_no_zero_dice = []
        for dice_5 in n_dice:
            if dice_5 == 0:
                if int(classes["nipple"][sample_i]) == 1:
                    n_no_zero_dice.append(dice_5)
                else:
                    pass
            else:
                n_no_zero_dice.append(dice_5)

    se_wb_batch_dice = np.std(wb_no_zero_dice) / math.sqrt(len(wb_no_zero_dice))
    se_b_batch_dice = np.std(b_no_zero_dice) / math.sqrt(len(b_no_zero_dice))
    se_m_batch_dice = np.std(m_no_zero_dice) / math.sqrt(len(m_no_zero_dice))
    se_sf_batch_dice = np.std(sf_no_zero_dice) / math.sqrt(len(sf_no_zero_dice))
    se_n_batch_dice = np.std(n_no_zero_dice) / math.sqrt(len(n_no_zero_dice))

    wb_batch_dice = np.array(wb_no_zero_dice).mean(axis=0)
    b_batch_dice = np.array(b_no_zero_dice).mean(axis=0)
    m_batch_dice = np.array(m_no_zero_dice).mean(axis=0)
    sf_batch_dice = np.array(sf_no_zero_dice).mean(axis=0)
    n_batch_dice = np.array(n_no_zero_dice).mean(axis=0)

    batch_standard_error = [
        se_wb_batch_dice,
        se_b_batch_dice,
        se_m_batch_dice,
        se_sf_batch_dice,
        se_n_batch_dice,
    ]

    batch_dice = [
        wb_batch_dice,
        b_batch_dice,
        m_batch_dice,
        sf_batch_dice,
        n_batch_dice,
    ]

    return batch_dice, batch_standard_error


def path_to_df_from_config(cfg):
    data_dir = pathlib.Path(cfg["data"]["dir"])

    img_paths = data_dir.glob(cfg["data"]["img_path"])
    img_sorted = sorted([x for x in img_paths])

    p_id_path = os.path.join(
        str(cfg["data"]["dir"]), str(cfg["data"]["patient_ids"].split("/")[-1])
    )
    patient_id = pd.read_csv(p_id_path)
    datafram = {"images": img_sorted, "id": patient_id.values.tolist()}
    df = pd.DataFrame(data=datafram)

    return df


def find_the_best_threshold(meta_probs, labels, batch):
    maximun_wb = 0.0
    best_threshold_wb = 0.0

    maximun_b = 0.0
    best_threshold_b = 0.0

    maximun_m = 0.0
    best_threshold_m = 0.0

    maximun_sf = 0.0
    best_threshold_sf = 0.0

    maximun_n = 0.0
    best_threshold_n = 0.0

    for i in list(np.arange(0.0, 1.0, 0.01)):
        # print(i)
        probs_for_threshold = (
            meta_probs.sigmoid().ge(i).to("cpu").float().numpy().astype(np.uint8)
        )

        batch_dice_for_threshold = dice_list(
            labels=labels, prediction=probs_for_threshold, classes=batch["classes_"]
        )

        if batch_dice_for_threshold[0] >= maximun_wb:
            maximun_wb = batch_dice_for_threshold[0]
            best_threshold_wb = i

        if batch_dice_for_threshold[1] >= maximun_b:
            maximun_b = batch_dice_for_threshold[1]
            best_threshold_b = i

        if batch_dice_for_threshold[2] >= maximun_m:
            maximun_m = batch_dice_for_threshold[2]
            best_threshold_m = i

        if batch_dice_for_threshold[3] >= maximun_sf:
            maximun_sf = batch_dice_for_threshold[3]
            best_threshold_sf = i

        if batch_dice_for_threshold[4] >= maximun_n:
            maximun_n = batch_dice_for_threshold[4]
            best_threshold_n = i

        best_threshold_list = [
            best_threshold_wb,
            best_threshold_b,
            best_threshold_m,
            best_threshold_sf,
            best_threshold_n,
        ]

    return best_threshold_list


# def threshold_from_batch(probs):
#
#     return thresholded
