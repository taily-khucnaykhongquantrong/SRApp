import matlab.engine
from imageio import imread

from utils.util import modcrop


def perceptual_index(srPath: str, hrPath: str):
    eng = matlab.engine.start_matlab()

    envPath = "utils/perceptual_index"
    eng.addpath(envPath, nargout=0)

    sr = imread(srPath)
    hr = imread(hrPath)

    cropped_hr = modcrop(hr, 4)
    sr = matlab.uint8(sr.tolist())
    cropped_hr = matlab.uint8(cropped_hr.tolist())

    score = eng.evaluate(sr, cropped_hr)

    eng.quit()

    return score
