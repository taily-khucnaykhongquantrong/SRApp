import numpy as np
from PIL import Image, ImageDraw

IMG_EXTENSION = ["jpg", "png", "jpeg"]


def modcrop(img_in, scale):
    """
    Make sure that the SR and HR have the same (H, W, C)
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

    for i, (metric_name, score) in enumerate(scoreList):
        drawnImg.text((10, 10 * i), metric_name + ": " + str(score), fill=(255, 87, 34))

    return img


def allowed_file(filename):
    """
    Return true if input image extension is in allowed list
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in IMG_EXTENSION
