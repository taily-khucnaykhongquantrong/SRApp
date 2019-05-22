import os
from glob import glob
from PIL import Image

# Scale factor
ratio = 4


def upscale(ratio, resample):
    sr_path = "img/sr/"
    hr_list = sorted(glob("img/hr/*"))

    for index, value in enumerate(hr_list):
        # Read image
        img = Image.open(value)
        dst = img.resize((tuple([int(x * ratio) for x in img.size])), resample)
        dst.save(sr_path + os.path.basename(value))


def bicubic():
    upscale(ratio, Image.BICUBIC)


def nearest():
    upscale(ratio, Image.NEAREST)


def bilinear():
    upscale(ratio, Image.BILINEAR)
