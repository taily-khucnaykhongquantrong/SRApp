from utils.ma_score import Mascore as ma
from utils.niqe import niqe


def perceptual_index(imgPath: str):
    maScore = ma(imgPath)
    niqeScore = niqe(imgPath)

    return ((10 - maScore) + niqeScore) / 2
