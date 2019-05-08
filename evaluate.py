from glob import glob
import os
from datetime import datetime
import subprocess

from utils.util import allowed_file
from utils.psnr import psnr
from utils.ssim import calculate_ssim as ssim
from utils.perceptual_index import perceptual_index
from utils.fid_score import calculate_fid_given_paths as fid
from utils.util import drawtext


def evaluate(model: str, isGif: bool):
    HR_DIRECTORY = "img/hr/*"
    SR_DIRECTORY = "img/sr/" + model + "/*"
    SR_WITH_SCORE_DIRECTORY = "img/sr_withscore/" + model + "/"

    hrList = sorted(glob(HR_DIRECTORY))
    srList = sorted(glob(SR_DIRECTORY))
    scoredImagesList = []

    print(str(datetime.now()))

    for (sr, hr) in zip(srList, hrList):
        fileNameSr = os.path.basename(sr)
        fileNameHr = os.path.basename(hr)

        if allowed_file(fileNameSr) and allowed_file(fileNameHr):
            scoreList = {}

            print(fileNameSr, fileNameHr, str(datetime.now()), sep=" | ")

            perceptualIndex = perceptual_index(sr, hr)
            scoreList["PI"] = round(perceptualIndex, 4)

            psnrValue = psnr(sr, hr)
            scoreList["PSNR"] = round(psnrValue, 4)

            ssimValue = ssim(sr, hr)
            scoreList["SSIM"] = round(ssimValue, 4)

            scoredImagesList.append(scoreList)

            img = drawtext(sr, scoreList)
            img.save(SR_WITH_SCORE_DIRECTORY + fileNameHr)

    fidValue = fid(
        [HR_DIRECTORY[:-1], SR_DIRECTORY[:-1]], batch_size=50, cuda=False, dims=2048
    )

    if isGif:
        subprocess.call(
            [
                "utils/Images2GIF/gifski.exe",
                "-o",
                "img/gif/" + model + ".gif",
                "--fps",
                "1",
                SR_WITH_SCORE_DIRECTORY + "*.png",
            ]
        )
        subprocess.call(
            [
                "utils/GIF2Video/ffmpeg.exe",
                "-i",
                "img/gif/" + model + ".gif",
                "-c",
                "vp9",
                "-crf",
                "0",
                "img/gif/" + model + ".webm",
            ]
        )

    print(str(datetime.now()))

    return scoredImagesList, fidValue
