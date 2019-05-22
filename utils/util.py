import os
import shutil
import logging
import subprocess
from glob import glob
from PIL import Image, ImageDraw

import numpy as np


def initLogger(fileName):
    logger = logging.getLogger("SRApp")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")

    fh = logging.FileHandler(fileName, "w")
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def modcrop(img_in, scale):
    """
    Make sure that the SR and HR have the same (H, W, C)
    Parameters
    ----------
    img_in: numpy array or image(H, W, C) or image(H, W).
            Image to get cropped.\n
    scale:  int.
            Scale factor.

    Returns
    -------
    img: ndarray.
         Return image as numpy array.
    """
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r, :]
    else:
        raise ValueError("Wrong img ndim: [{:d}].".format(img.ndim))
    return img


def drawtext(imgPath: str, scoreList):
    """
    Open and draw score on the image.

    Parameters
    ----------
    imgPath: str
        Path to the image.\n
    args: list
        List of sr metric scores.

    Returns
    -------
    img: PIL.Image.Image
        Image class from Pillow, save it later by img.save() method.
    """
    img = Image.open(imgPath)
    drawnImg = ImageDraw.Draw(img)

    for i, (key, score) in enumerate(scoreList.items()):
        drawnImg.text((10, 10 * i), key + ": " + str(score), fill=(255, 87, 34))

    return img


def allowed_file(filename, IMG_EXTENSION=["jpg", "png", "jpeg"]):
    """
    Return true if input image extension is in allowed list
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in IMG_EXTENSION


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def cleanDir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def downscale(lr_path: str, hr_path: str, scale=4):
    hr_list = sorted(glob(hr_path))
    # Scale factor
    ratio = 1 / scale

    for index, value in enumerate(hr_list):
        # Read image
        img = Image.open(value)
        dst = img.resize((tuple([int(x * ratio) for x in img.size])), Image.BICUBIC)
        dst.save(lr_path + os.path.basename(value))


def genVideo(path: str, model: str):
    imgList = glob(path + "*")

    if len(imgList) > 1:
        subprocess.call(
            [
                "utils/Images2GIF/gifski.exe",
                "-o",
                "img/gif/" + model + ".gif",
                "--fps",
                "1",
                path + "*.png",
            ]
        )
        subprocess.call(
            [
                "utils/GIF2Video/ffmpeg.exe",
                "-y",
                "-i",
                "img/gif/" + model + ".gif",
                "-c",
                "vp9",
                "-crf",
                "0",
                "img/gif/" + model + ".webm",
            ]
        )
    else:
        fileExtension = os.path.basename(imgList[0]).split(".")[1]
        shutil.copyfile(imgList[0], "img/sr/" + model + "." + fileExtension)
