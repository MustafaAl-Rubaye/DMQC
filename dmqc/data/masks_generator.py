import numpy as np
import cv2
import collections

file = "/home/mustafa/Documents/Mamography/data/merged_file.json"


def resized_masks(mask):

    old_dim = mask.shape
    old_H = old_dim[0]
    old_W = old_dim[1]

    h = 512
    w = 256

    if old_W < old_H // 2:
        delta = old_H // 2 - old_W
        top, bottom = 0, 0
        right, left = delta // 2, delta // 2

        color = [0, 0, 0]
        new_ = cv2.copyMakeBorder(
            src=mask,
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
        new_ = cv2.copyMakeBorder(
            src=mask,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=color,
        )

    mask_resized = cv2.resize(new_, (w, h))

    return mask_resized


def XY_to_list(list_points):
    flatten_list_points = sum(list_points, [])

    x = 0
    new_list = []
    while x < len(flatten_list_points):
        new_list.append(flatten_list_points[x : x + 2])
        x += 2

    return new_list


def fill_mask_poly(new_list, tmp):

    np_points = np.array(new_list)[None, ...].astype(np.int32)
    mask = cv2.fillPoly(tmp, np_points, 1)

    return mask


def get_masks_from_json(ori_data, wanted):
    for data_i in ori_data:
        n_images = len(data_i["images"])

        for image_i in range(n_images):
            image_name = data_i["images"][image_i]["file_name"]

            if image_name == wanted:
                H = data_i["images"][image_i]["height"]
                W = data_i["images"][image_i]["width"]

                # this is the main order of the classes
                tmp1 = np.zeros((H, W), dtype=np.uint8)
                tmp2 = np.zeros((H, W), dtype=np.uint8)
                tmp3 = np.zeros((H, W), dtype=np.uint8)
                tmp4 = np.zeros((H, W), dtype=np.uint8)
                tmp5 = np.zeros((H, W), dtype=np.uint8)
                classes_dic_ = {
                    "whole breast": tmp5,
                    "breast": tmp1,
                    "muscle": tmp3,
                    "skin folding": tmp2,
                    "nipple": tmp4,
                }

                classes = {
                    "whole breast": 0,
                    "breast": 0,
                    "muscle": 0,
                    "skin folding": 0,
                    "nipple": 0,
                }

                classes_dic = collections.OrderedDict(classes_dic_)

                n_annotations = len(data_i["annotations"])
                for ids in range(n_annotations):
                    image_json_id = data_i["annotations"][ids]["image_id"]

                    list_points = data_i["annotations"][ids]["segmentation"]
                    new_list = XY_to_list(
                        list_points
                    )  # [x,y,x,y,x,y] to [[x,y],[x,y],.....]

                    class_title = data_i["annotations"][ids]["category_id"]

                    if (
                        image_json_id == image_i
                    ):  # get the annotations from the img ID from the annnotation list
                        # mask = fill_mask_poly(new_list, classes_dic_)
                        if class_title == 1:
                            classes_dic_["breast"] = fill_mask_poly(
                                new_list, classes_dic_["breast"]
                            )
                            classes["breast"] = 1

                        elif class_title == 2:
                            classes_dic_["muscle"] = fill_mask_poly(
                                new_list, classes_dic_["muscle"]
                            )
                            classes["muscle"] = 1
                        elif class_title == 3:
                            classes_dic_["nipple"] = fill_mask_poly(
                                new_list, classes_dic_["nipple"]
                            )
                            classes["nipple"] = 1
                        elif class_title == 4:
                            classes_dic_["skin folding"] = fill_mask_poly(
                                new_list, classes_dic_["skin folding"]
                            )
                            classes["skin folding"] = 1
                        elif class_title == 5:
                            classes_dic_["whole breast"] = fill_mask_poly(
                                new_list, classes_dic_["whole breast"]
                            )
                            classes["whole breast"] = 1

                    # for key, value in classes_dic_.items():
                    #     classes_dic_[key] = resized_masks(classes_dic[value])
                    # del classes_dic[key]
                    # value.replace(None, tmp)

                dic_array = list(classes_dic.values())

                mask_list = []
                for a_mask in dic_array:
                    mask_list.append(resized_masks(a_mask))

                # dict_list = []
                # for mask in dic_array:
                #     mask.squeeze()
                #     dict_list.append(mask)

    return mask_list, classes
