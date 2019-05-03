from glob import glob
import os
import itertools

from utils.util import allowed_file

HR_DIRECTORY = 'img/hr/*'
SR_DIRECTORY = 'img/sr/*'
SR_WITH_SCORE_DIRECTORY = 'img/sr_withscore/*'

hrList = sorted(glob(HR_DIRECTORY))
srList = sorted(glob(SR_DIRECTORY))

for (sr, hr) in itertools.product(srList, hrList):
    fileNameSr = os.path.basename(sr)
    fileNameHr = os.path.basename(hr)

    if (allowed_file(fileNameSr) and allowed_file(fileNameHr)):
        print(fileNameSr, fileNameHr, sep=" | ")
