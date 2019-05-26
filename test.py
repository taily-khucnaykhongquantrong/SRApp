from models.esrgan.test import test as esrgan
from models.high2low.test import test as high2low
from models.non_ML.test import bicubic, nearest, bilinear


def test():
    esrgan()
    high2low()
    bicubic()
    nearest()
    bilinear()
