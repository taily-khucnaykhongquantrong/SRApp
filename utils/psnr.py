import math
import numpy as np
import imageio

from utils.util import modcrop


def psnr(imgPath1, imgPath2):
    img1 = imageio.imread(imgPath1)
    img2 = imageio.imread(imgPath2)

    img1 = modcrop(img1, 4)
    img2 = modcrop(img2, 4)

    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))
