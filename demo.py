from glob import glob
import os
import matlab.engine

from utils.util import allowed_file
from utils.perceptualIndex import perceptual_index
from utils.psnr import psnr
from utils.ssim import calculate_ssim as ssim
from utils.util import drawtext

HR_DIRECTORY = 'img/hr/*'
SR_DIRECTORY = 'img/sr/*'
SR_WITH_SCORE_DIRECTORY = 'img/sr_withscore/'

hrList = sorted(glob(HR_DIRECTORY))
srList = sorted(glob(SR_DIRECTORY))
engine = matlab.engine.start_matlab()

for (sr, hr) in zip(srList, hrList):
    fileNameSr = os.path.basename(sr)
    fileNameHr = os.path.basename(hr)

    if (allowed_file(fileNameSr) and allowed_file(fileNameHr)):
        scoreList = []

        print(sr, hr, sep=" | ")

        perceptualIndex = perceptual_index(sr, engine)
        scoreList.append(('PI', round(perceptualIndex, 4)))

        psnrValue = psnr(sr, hr)
        scoreList.append(('PSNR', round(psnrValue, 4)))

        ssimValue = ssim(sr, hr)
        scoreList.append(('SSIM', round(ssimValue, 4)))

        img = drawtext(sr, scoreList)
        img.save(SR_WITH_SCORE_DIRECTORY + fileNameHr)

engine.quit()
