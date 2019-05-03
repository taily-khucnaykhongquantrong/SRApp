from utils.ma_score import Mascore as ma
from utils.niqe import niqe


def perceptual_index(imgPath: str, eng):
    maScore = ma(imgPath, eng)
    niqeScore = niqe(imgPath, eng)

    return ((10 - maScore) + niqeScore) / 2
