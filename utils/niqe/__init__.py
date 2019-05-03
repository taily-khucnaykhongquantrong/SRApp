import matlab.engine
from PIL import Image


def niqe(imgPath: str, eng):
    envPath = "utils/niqe"
    eng.addpath(envPath, nargout=0)

    img = Image.open(imgPath)
    img_mat = matlab.uint8(list(img.getdata()))
    # This results an image with order [W, H, C]
    img_mat.reshape((img.size[0], img.size[1], 3))

    # Permute to image with order [H, W, C]
    # Matlab imread only as order [H, W, C]
    img_mat = eng.permute(img_mat, matlab.uint8([2, 1, 3]))
    score = eng.niqe(img_mat)

    return score
