import cv2
import numpy as np
import argparse
import glob
import json
import matplotlib.pyplot as plt
from statistics import mean
from skimage.transform import hough_line, hough_line_peaks

from sklearn.linear_model import LinearRegression
from matplotlib import cm
import numpy.linalg as la


def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def muscle_nipple_line(muscle, nipple):

    if nipple.max() != 1:
        print("- The nipple is not showing in the image")

    else:
        print("- The nipple is in profile")

        x, y = muscle.shape
        muscle_i_list = []
        # muscle_j_list = []

        nipple_i_list = []
        # nipple_j_list = []
        for i in range(x):
            for j in range(y):
                if muscle[i, j] == 1:
                    muscle_i_list.append(i)
                    # muscle_j_list.append(j)

                if nipple[i, j] == 1:
                    nipple_i_list.append(j)
                    # nipple_j_list.append(j)

        if max(nipple_i_list) < min(muscle_i_list):
            print("- The pectoral muscle is NOT sufficiently long")

        else:
            print("- The pectoral muscle is sufficiently long: shadow to nipple level")

    return


def Extract(lst):
    x = [item[0] for item in lst]
    y = [item[1] for item in lst]

    return x, y


def muscle_angle(muscle):
    shape = muscle.shape
    x_shape = shape[0]
    y_shape = shape[1]
    middle_list = []
    edge_list = []
    j_list = []

    edges = cv2.Canny(np.uint8(muscle), 1, 0)

    for i in range(x_shape):
        for j in range(y_shape):
            if edges[i, j] != 0:
                j_list.append([i, j])

    for i in range(x_shape):
        for j in range(y_shape):
            if edges[i, j] != 0:
                # iter_ += 1
                if i > (j_list[0][0] + 5):
                    middle_list.append([i, j])
                # j_list.append(j)

    for i in range(x_shape):
        for j in range(y_shape):
            if edges[i, j] != 0:
                # iter_ += 1
                if i > (j_list[0][0] + 5):
                    edge_list.append([i, j])

    for i_x in range(len(edge_list)):
        edge_list[i_x][1] = 0

    for i in edge_list:
        edges[i[0], i[1]] = 255

    x, y = Extract(middle_list)

    x_2, y_2 = Extract(middle_list)

    reg_1 = LinearRegression()
    reg_1.fit(middle_list, y)

    reg_2 = LinearRegression()
    reg_2.fit(edge_list, y_2)

    # plt.plot(middle_list, reg_1.predict(middle_list), color='k')
    # plt.plot(edge_list, reg_2.predict(edge_list), color='k')
    # plt.show()

    # Constructing test image

    ################################### TODO:
    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    # h, theta, d = hough_line(edges, theta=tested_angles)
    #
    # # Generating figure 1
    # fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    # ax = axes.ravel()
    #
    # ax[0].imshow(edges, cmap=cm.gray)
    # ax[0].set_title('Input image')
    # ax[0].set_axis_off()
    #
    # ax[1].imshow(np.log(1 + h),
    #              extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
    #              cmap=cm.gray, aspect=1 / 1.5)
    # ax[1].set_title('Hough transform')
    # ax[1].set_xlabel('Angles (degrees)')
    # ax[1].set_ylabel('Distance (pixels)')
    # ax[1].axis('image')
    #
    # ax[2].imshow(edges, cmap=cm.gray)
    # origin = np.array((0, edges.shape[1]))
    # for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    #     y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    #     ax[2].plot(origin, (y0, y1), '-r')
    # ax[2].set_xlim(origin)
    # ax[2].set_ylim((edges.shape[0], 0))
    # ax[2].set_axis_off()
    # ax[2].set_title('Detected lines')
    #
    # plt.tight_layout()
    # plt.show()

    ################################### TODO: 2

    h, theta, d = hough_line(edges)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    ax = axes.ravel()

    ax[0].imshow(edges, cmap=cm.gray)
    ax[0].set_title("Input image")
    ax[0].set_axis_off()
    # ax[1].imshow(np.log(1 + h),
    #              extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
    #              cmap=cm.gray, aspect=1 / 1.5)
    # ax[1].set_title('Hough transform')
    # ax[1].set_xlabel('Angles (degrees)')
    # ax[1].set_ylabel('Distance (pixels)')
    # ax[1].axis('image')

    ax[1].imshow(edges, cmap=cm.gray)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - edges.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[1].plot((0, edges.shape[1]), (y0, y1), "-r")
    ax[1].set_xlim((0, edges.shape[1]))
    ax[1].set_ylim((edges.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title("Detected lines")

    plt.tight_layout()
    plt.show()

    angle = []
    dist = []
    for _, a, d in zip(*hough_line_peaks(h, theta, d)):
        angle.append(a)
        dist.append(d)

    angle = [a * 180 / np.pi for a in angle]
    angle_reel = np.max(angle) - np.min(angle)

    print(f"- The muscle angle is {round(angle_reel, 2)}")

    return


# def best_fit_slope(xs, ys):
#     m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) ** 2) - mean(xs ** 2))
#     return m


def best_fit_slope(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / (
        (mean(xs) * mean(xs)) - mean(xs * xs)
    )
    return m


def artifacts(skinfolds, nipple):

    if skinfolds.max() == 0:
        print("- There is no skinfolds in the image")

    else:
        print("- There is skinfolds in the image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/6_mod_300_epo/masks_json/87.json",
    )
    args = parser.parse_args()

    # for file_path in glob.glob(args.root):
    with open(args.root) as json_file:
        data = json.load(json_file)
        wb_img = np.squeeze(data["whole breast"])
        b_img = np.squeeze(data["breast"])
        m_img = np.squeeze(data["muscle"])
        sf_img = np.squeeze(data["skinfolds"])
        n_img = np.squeeze(data["nipple"])

        name = json_file.name.split("/")[-1][:-5]

        print(f"Image number is {name}")
        muscle_nipple_line(m_img, n_img)
        artifacts(sf_img, n_img)

        muscle_angle(m_img)

        # plt.imshow(wb_img, cmap='gray')
        # plt.imshow(b_img, cmap='gray')
        # plt.imshow(m_img, cmap='gray')
        # plt.imshow(sf_img, cmap='gray')
        # plt.imshow(n_img, cmap='gray')
        # plt.show()
