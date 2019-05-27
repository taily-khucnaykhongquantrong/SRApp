from glob import glob
import os
from datetime import datetime

from utils.util import allowed_file
from utils.psnr import psnr
from utils.ssim import calculate_ssim as ssim
from utils.perceptual_index import perceptual_index
from utils.fid_score import calculate_fid_given_paths as fid
from utils.util import drawtext, initLogger, genVideo


def evaluate(model: str) -> (list, float):
    HR_DIRECTORY = "img/hr/*"
    SR_DIRECTORY = "img/sr/" + model + "/*"
    SR_WITH_SCORE_DIRECTORY = "img/sr_withscore/" + model + "/"

    hrList = sorted(glob(HR_DIRECTORY))
    srList = sorted(glob(SR_DIRECTORY))
    scoredImagesList = []
    logger = initLogger("evaluated_" + model + ".log")
    piSum = 0
    psnrSum = 0
    ssimSum = 0

    for (sr, hr) in zip(srList, hrList):
        fileNameSr = os.path.basename(sr)
        fileNameHr = os.path.basename(hr)

        if allowed_file(fileNameSr) and allowed_file(fileNameHr):
            scoreList = {}

            perceptualIndex = perceptual_index(sr, hr)
            scoreList["PI"] = round(perceptualIndex, 4)

            psnrValue = psnr(sr, hr)
            scoreList["PSNR"] = round(psnrValue, 4)

            ssimValue = ssim(sr, hr)
            scoreList["SSIM"] = round(ssimValue, 4)

            logger.info(
                "Filename: "
                + fileNameSr
                + ", PI: "
                + str(scoreList["PI"])
                + ", PSNR: "
                + str(scoreList["PSNR"])
                + ", SSIM: "
                + str(scoreList["SSIM"])
            )

            piSum += scoreList["PI"]
	    psnrSum += scoreList["PSNR"]
	    ssimSum += scoreList["SSIM"]

            scoredImagesList.append(scoreList)

            img = drawtext(sr, scoreList)
            img.save(SR_WITH_SCORE_DIRECTORY + fileNameHr)

    fidValue = fid(
        [HR_DIRECTORY[:-1], SR_DIRECTORY[:-1]], batch_size=30, cuda=False, dims=2048
    )
    totalImage = len(srList)
    averagePI = piSum / totalImage
    averagePSNR = psnrSum / totalImage
    averageSSIM = ssimSum / totalImage

    genVideo(SR_WITH_SCORE_DIRECTORY, model)
    logger.info("FID: " + str(fidValue))
    logger.info("Average PI: " + str(averagePI))
    logger.info("Average PSNR: " + str(averagePSNR))
    logger.info("Average SSIM: " + str(averageSSIM))

    return scoredImagesList, fidValue
