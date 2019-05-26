import os
from glob import glob
from PIL import Image

from utils.util import cleanDir

# Scale factor
ratio = 4


def upscale(ratio, resample, interpName):
    sr_path = "img/sr/" + interpName + "/"
    lr_list = sorted(glob("img/lr/*"))

    cleanDir(sr_path)

    for index, value in enumerate(lr_list):
        # Read image
        img = Image.open(value)
        dst = img.resize((tuple([int(x * ratio) for x in img.size])), resample)
        dst.save(sr_path + os.path.basename(value))


def bicubic():
    upscale(ratio, Image.BICUBIC, 'bicubic')


def nearest():
    upscale(ratio, Image.NEAREST, 'nearest')


def bilinear():
    upscale(ratio, Image.BILINEAR, 'bilinear')
