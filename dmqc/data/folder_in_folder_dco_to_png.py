import pydicom as dicom
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import argparse
import glob
import pandas as pd
from tqdm import tqdm
import random
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def apply_voi_lut(arr, ds, index=0):
    """Apply a VOI lookup table or windowing operation to `arr`.
    .. versionadded:: 1.4
    Parameters
    ----------
    arr : numpy.ndarray
        The :class:`~numpy.ndarray` to apply the VOI LUT or windowing operation
        to.
    ds : dataset.Dataset
        A dataset containing a :dcm:`VOI LUT Module<part03/sect_C.11.2.html>`.
        If (0028,3010) *VOI LUT Sequence* is present then returns an array
        of ``np.uint8`` or ``np.uint16``, depending on the 3rd value of
        (0028,3002) *LUT Descriptor*. If (0028,1050) *Window Center* and
        (0028,1051) *Window Width* are present then returns an array of
        ``np.float64``. If neither are present then `arr` will be returned
        unchanged.
    index : int, optional
        Where the VOI LUT Module contains multiple possible views, this is
        the index of the view to return (default ``0``).
    Returns
    -------
    numpy.ndarray
        An array with applied VOI LUT or windowing operation.
    Notes
    -----
    When the dataset requires a modality LUT or rescale operation as part of
    the Modality LUT module then that must be applied before any windowing
    operation.
    See Also
    --------
    :func:`~pydicom.pixel_data_handlers.util.apply_modality_lut`
    References
    ----------
    * DICOM Standard, Part 3, :dcm:`Annex C.11.2
      <part03/sect_C.11.html#sect_C.11.2>`
    * DICOM Standard, Part 3, :dcm:`Annex C.8.11.3.1.5
      <part03/sect_C.8.11.3.html#sect_C.8.11.3.1.5>`
    * DICOM Standard, Part 4, :dcm:`Annex N.2.1.1
      <part04/sect_N.2.html#sect_N.2.1.1>`
    """
    if "VOILUTSequence" in ds:
        # VOI LUT Sequence contains one or more items
        item = ds.VOILUTSequence[index]
        nr_entries = item.LUTDescriptor[0] or 2 ** 16
        first_map = item.LUTDescriptor[1]

        # PS3.3 C.8.11.3.1.5: may be 8, 10-16
        nominal_depth = item.LUTDescriptor[2]
        if nominal_depth in list(range(10, 17)):
            dtype = "uint16"
        elif nominal_depth == 8:
            dtype = "uint8"
        else:
            raise NotImplementedError(
                "'{}' bits per LUT entry is not supported".format(nominal_depth)
            )

        lut_data = np.asarray(item.LUTData, dtype=dtype)

        # IVs < `first_map` get set to first LUT entry (i.e. index 0)
        clipped_iv = np.zeros(arr.shape, dtype=arr.dtype)
        # IVs >= `first_map` are mapped by the VOI LUT
        # `first_map` may be negative, positive or 0
        mapped_pixels = arr >= first_map
        clipped_iv[mapped_pixels] = arr[mapped_pixels] - first_map
        # IVs > number of entries get set to last entry
        np.clip(clipped_iv, 0, nr_entries - 1, out=clipped_iv)

        return lut_data[clipped_iv]
    elif "WindowCenter" in ds and "WindowWidth" in ds:
        if ds.PhotometricInterpretation not in ["MONOCHROME1", "MONOCHROME2"]:
            raise ValueError(
                "When performing a windowing operation only 'MONOCHROME1' and "
                "'MONOCHROME2' are allowed for (0028,0004) Photometric "
                "Interpretation"
            )

        # May be LINEAR (default), LINEAR_EXACT, SIGMOID or not present, VM 1
        voi_func = getattr(ds, "VOILUTFunction", "LINEAR").upper()
        # VR DS, VM 1-n
        elem = ds["WindowCenter"]
        center = elem.value[index] if elem.VM > 1 else elem.value
        elem = ds["WindowWidth"]
        width = elem.value[index] if elem.VM > 1 else elem.value

        # The output range depends on whether or not a modality LUT or rescale
        #   operation has been applied
        if "ModalityLUTSequence" in ds:
            # Unsigned - see PS3.3 C.11.1.1.1
            y_min = 0
            bit_depth = ds.ModalityLUTSequence[0].LUTDescriptor[2]
            y_max = 2 ** bit_depth - 1
        elif ds.PixelRepresentation == 0:
            # Unsigned
            y_min = 0
            y_max = 2 ** ds.BitsStored - 1
        else:
            # Signed
            y_min = -(2 ** (ds.BitsStored - 1))
            y_max = 2 ** (ds.BitsStored - 1) - 1

        if "RescaleSlope" in ds and "RescaleIntercept" in ds:
            # Otherwise its the actual data_ range
            y_min = y_min * ds.RescaleSlope + ds.RescaleIntercept
            y_max = y_max * ds.RescaleSlope + ds.RescaleIntercept

        y_range = y_max - y_min
        arr = arr.astype("float64")

        if voi_func in ["LINEAR", "LINEAR_EXACT"]:
            # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
            if voi_func == "LINEAR":
                if width < 1:
                    raise ValueError(
                        "The (0028,1051) Window Width must be greater than or "
                        "equal to 1 for a 'LINEAR' windowing operation"
                    )
                center -= 0.5
                width -= 1
            elif width <= 0:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than 0 "
                    "for a 'LINEAR_EXACT' windowing operation"
                )

            below = arr <= (center - width / 2)
            above = arr > (center + width / 2)
            between = np.logical_and(~below, ~above)

            arr[below] = y_min
            arr[above] = y_max
            if between.any():
                arr[between] = ((arr[between] - center) / width + 0.5) * y_range + y_min
        elif voi_func == "SIGMOID":
            # PS3.3 C.11.2.1.3.1
            if width <= 0:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than 0 "
                    "for a 'SIGMOID' windowing operation"
                )

            arr = y_range / (1 + np.exp(-4 * (arr - center) / width)) + y_min
        else:
            raise ValueError(
                "Unsupported (0028,1056) VOI LUT Function value '{}'".format(voi_func)
            )

    return arr


def apply_modality_lut(arr, ds):
    """Apply a modality lookup table or rescale operation to `arr`.
    .. versionadded:: 1.4
    Parameters
    ----------
    arr : numpy.ndarray
        The :class:`~numpy.ndarray` to apply the modality LUT or rescale
        operation to.
    ds : dataset.Dataset
        A dataset containing a :dcm:`Modality LUT Module
        <part03/sect_C.11.html#sect_C.11.1>`.
    Returns
    -------
    numpy.ndarray
        An array with applied modality LUT or rescale operation. If
        (0028,3000) *Modality LUT Sequence* is present then returns an array
        of ``np.uint8`` or ``np.uint16``, depending on the 3rd value of
        (0028,3002) *LUT Descriptor*. If (0028,1052) *Rescale Intercept* and
        (0028,1053) *Rescale Slope* are present then returns an array of
        ``np.float64``. If neither are present then `arr` will be returned
        unchanged.
    Notes
    -----
    When *Rescale Slope* and *Rescale Intercept* are used, the output range
    is from (min. pixel value * Rescale Slope + Rescale Intercept) to
    (max. pixel value * Rescale Slope + Rescale Intercept), where min. and
    max. pixel value are determined from (0028,0101) *Bits Stored* and
    (0028,0103) *Pixel Representation*.
    References
    ----------
    * DICOM Standard, Part 3, :dcm:`Annex C.11.1
      <part03/sect_C.11.html#sect_C.11.1>`
    * DICOM Standard, Part 4, :dcm:`Annex N.2.1.1
      <part04/sect_N.2.html#sect_N.2.1.1>`
    """
    if "ModalityLUTSequence" in ds:
        item = ds.ModalityLUTSequence[0]
        nr_entries = item.LUTDescriptor[0] or 2 ** 16
        first_map = item.LUTDescriptor[1]
        nominal_depth = item.LUTDescriptor[2]

        dtype = "uint{}".format(nominal_depth)
        lut_data = np.asarray(item.LUTData, dtype=dtype)

        # IVs < `first_map` get set to first LUT entry (i.e. index 0)
        clipped_iv = np.zeros(arr.shape, dtype=arr.dtype)
        # IVs >= `first_map` are mapped by the Modality LUT
        # `first_map` may be negative, positive or 0
        mapped_pixels = arr >= first_map
        clipped_iv[mapped_pixels] = arr[mapped_pixels] - first_map
        # IVs > number of entries get set to last entry
        np.clip(clipped_iv, 0, nr_entries - 1, out=clipped_iv)

        return lut_data[clipped_iv]
    elif "RescaleSlope" in ds and "RescaleIntercept" in ds:
        arr = arr.astype(np.float64) * ds.RescaleSlope
        arr += ds.RescaleIntercept

    return arr


def dicom_img_spacing(data):
    spacing = None

    for spacing_param in [
        "Imager Pixel Spacing",
        "ImagerPixelSpacing",
        "PixelSpacing",
        "Pixel Spacing",
    ]:
        if hasattr(data, spacing_param):
            spacing_attr_value = getattr(data, spacing_param)
            if isinstance(spacing_attr_value, str):
                if isfloat(spacing_attr_value):
                    spacing = float(spacing_attr_value)
                else:
                    spacing = float(spacing_attr_value.split()[0])
            elif isinstance(spacing_attr_value, dicom.multival.MultiValue):
                if len(spacing_attr_value) != 2:
                    return None
                spacing = list(map(lambda x: float(x), spacing_attr_value))[0]
            elif isinstance(spacing_attr_value, float):
                spacing = spacing_attr_value
        else:
            continue

        if spacing is not None:
            break
    return spacing


def read_dicom(filename, spacing_none_mode=True):
    """
    Reads a dicom file
    Parameters
    ----------
    filename : str or pydicom.dataset.FileDataset
        Full path to the image
    spacing_none_mode: bool
        Whether to return None if spacing info is not present. When False the output of the function
        will be None only if there are any issues with the image.
    Returns
    -------
    out : tuple
        Image itself as uint16, spacing, and the DICOM metadata
    """

    if isinstance(filename, str):
        try:
            data = dicom.read_file(filename)
        except:
            raise UserWarning("Failed to read the dicom.")
            return None
    elif isinstance(filename, dicom.dataset.FileDataset):
        data = filename
    else:
        raise TypeError(
            "Unknown type of the filename. Mightbe either string or pydicom.dataset.FileDataset."
        )

    img = np.frombuffer(data.PixelData, dtype=np.uint16).copy().astype(np.float64)

    if data.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img
    try:
        img = img.reshape((data.Rows, data.Columns))
    except:
        raise UserWarning("Could not reshape the image while reading!")
        return None

    spacing = dicom_img_spacing(data)
    img = img.astype(np.uint16)
    if spacing_none_mode:
        if spacing is not None:
            return img, spacing, data
        else:
            raise UserWarning("Could not read the spacing information!")
            return None

    return img, spacing, data


def process_xray(img, cut_min=5, cut_max=99, multiplier=255):
    # This function changes the histogram of the image by doing global contrast normalization
    # cut_min - lowest percentile which is used to cut the image histogram
    # cut_max - highest percentile

    img = img.copy()
    lim1, lim2 = np.percentile(img, [cut_min, cut_max])
    img[img < lim1] = lim1
    img[img > lim2] = lim2

    img -= lim1
    img /= img.max()
    img *= multiplier

    return img


def Histograms_Equalization(img):
    hist, bins = np.histogram(img.flatten(), 65535, [0, 65535])
    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint16")

    img = cdf[img]

    return img


if __name__ == "__main__":
    # --dicom_root /media/mustafa/"My Passport"/cp2157
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_root", default="/media/mustafa/My Passport/cp2157/*")
    parser.add_argument(
        "--save_dir", default="/home/mustafa/Documents/Mammo_project/data/"
    )
    parser.add_argument("--n_samples", type=int, default=5000)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.dicom_root, "*/*.dcm"))

    os.makedirs(os.path.join(args.save_dir, "processed"), exist_ok=True)

    files = random.sample(files, len(files))

    number_of_samples = 0
    meta_info = []
    for idx, file in enumerate(tqdm(files, total=len(files))):
        try:
            # Read the data_ as uint16
            img, s, data = read_dicom(file)

        except UserWarning:
            continue

        try:

            if data.ViewPosition != "MLO":
                continue
            # try:
            # Apply LUT
            # img = apply_voi_lut(img, data).astype(np.float64)
            # except ValueError:
            #
            #     continue
            # Convert to 8- bit
            # img = process_xray(img, 0, 100, 255).astype(np.uint8)
            # # Apply pre-processing
            # img = cv2.equalizeHist(img)

            # cv2.imwrite(os.path.join(args.save_dir, "processed", f"{idx}.png"), img)
            meta_info.append(
                {
                    "Fname": file.replace(args.dicom_root, ""),
                    "ID": idx,
                    "Side": data.ImageLaterality,
                    "View": data.ViewPosition,
                    "patient_id": data.PatientID,
                }
            )
            number_of_samples += 1
            if number_of_samples == args.n_samples:
                break

            # plt.imshow(img, cmap=plt.cm.gray)
            # plt.show()

        except AttributeError:
            print("MLO was not found!!")
            pass

        except UserWarning:
            continue

    df = pd.DataFrame(data=meta_info)
    df.to_csv(os.path.join(args.save_dir, "metadata_processed.csv"), index=None)
