import matlab.engine
from PIL import Image


def Mascore(imgPath: str):
    eng = matlab.engine.start_matlab()
    envPath = "utils/ma_score"
    eng.addpath(envPath, nargout=0)

    img = Image.open(imgPath)
    img_mat = matlab.uint8(list(img.getdata()))
    img_mat.reshape((img.size[0], img.size[1], 3))
    score = eng.quality_predict(img_mat)

    eng.quit()

    return score
