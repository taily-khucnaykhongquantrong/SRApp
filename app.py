import os
from glob import glob

# import sys

from flask import (
    Flask,
    render_template,
    # flash,
    request,
    # redirect,
    # url_for,
    send_from_directory,
)
from werkzeug.utils import secure_filename

# import face_recognition

from test import test
from evaluate import evaluate
import utils.util as util

LR_FOLDER = "img/lr/"
HR_FOLDER = "img/hr/"
SR_ESRGAN_FOLDER = "img/sr/esrgan/"
SR_HIGH2LOW_FOLDER = "img/sr/high2low/"
GIF_FOLDER = "img/gif/*"
SECRET_KEY = b'_5#y2L"F4Q8z\n\xec]/'

app = Flask(__name__)
# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = SECRET_KEY


@app.route("/")
def index():
    # Extract only webm file type
    gifImages = [path for path in sorted(glob(GIF_FOLDER)) if path.endswith(".webm")]
    return render_template("index.html", gifImages=gifImages)


@app.route("/result", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        # Get the name of the uploaded files
        uploaded_files = request.files.getlist("file[]")
        isEvaluate = request.form.getlist("evaluate")
        isNeedHR = request.form.getlist("needHR")
        resultSlides = []
        filenames = []

        util.cleanDir(LR_FOLDER)
        util.cleanDir(SR_ESRGAN_FOLDER)
        util.cleanDir(SR_HIGH2LOW_FOLDER)
        if isNeedHR:
            util.cleanDir(HR_FOLDER)

        for file in uploaded_files:

            # Check if the file is one of the allowed types/extensions
            if file and util.allowed_file(file.filename):
                # Make the filename safe, remove unsupported chars
                filename = secure_filename(file.filename)
                # Move the file form the temporal folder to the upload
                # folder we setup
                path = HR_FOLDER if isNeedHR else LR_FOLDER
                file.save(os.path.join(path, filename))
                # Save the filename into a list, we'll use it later
                filenames.append(filename)
                # Redirect the user to the uploaded_file route, which
                # will basicaly show on the browser the uploaded file

        if isNeedHR:
            util.downscale(LR_FOLDER, HR_FOLDER + "*", 4)

        test()

        if isEvaluate:
            esrganScores, esrganFID = evaluate("esrgan")
            high2lowScores, high2lowFID = evaluate("high2low")
            bicubicScores, bicubicFID = evaluate("bicubic")
            nearstScores, nearestFID = evaluate("nearest")
            bilinearScores, bilinearFID = evaluate("bilinear")
        else:
            util.genVideo(SR_ESRGAN_FOLDER, 'esrgan')
            util.genVideo(SR_HIGH2LOW_FOLDER, 'high2low')
            esrganScores, esrganFID = (["N/A"], "N/A")
            high2lowScores, high2lowFID = (["N/A"], "N/A")
        util.genVideo('img/hr/', 'hr')

        resultSlides = [
            path for path in sorted(glob(GIF_FOLDER)) if path.endswith(".webm")
        ]
        scoresList = [
            (esrganScores, esrganFID, "ESRGAN"),
            (high2lowScores, high2lowFID, "High2Low"),
            (bicubicScores, bicubicFID, "Bicubic"),
            (nearstScores, nearestFID, "Nearest"),
            (bilinearScores, bilinearFID, "Bilinear")
        ]

        return render_template("result.html", results=zip(scoresList, resultSlides))

        # lrImages = []
        # lrImagesEncoding = []
        # hrImages = []
        # hrImagesEncoding = []
        # faceDistancesList = []
        # faceConfidenceList = []

        # i = 0

        # for file in lrFiles:
        #     lrImages.append(
        #         face_recognition.load_image_file(UPLOAD_FOLDER + lrFiles[i])
        #     )
        #     lrImagesEncoding.append(face_recognition.face_encodings(lrImages[i])[0])
        #     hrImages.append(
        #         face_recognition.load_image_file(RESULT_FOLDER + hrFiles[i])
        #     )
        #     hrImagesEncoding.append(face_recognition.face_encodings(hrImages[i])[0])

        #     faceDistancesList.append(
        #         face_recognition.face_distance(
        #             [lrImagesEncoding[i]], hrImagesEncoding[i]
        #         )
        #     )

        # faceConfidenceList.append(face_distance_to_conf(faceDistancesList[i]))

        # i = i + 1


@app.route("/video/<path:filePath>")
def getVideo(filePath):
    fileName = os.path.basename(filePath)
    fileDirectory = filePath.split(fileName)[0]

    return send_from_directory(fileDirectory, fileName)


@app.route("/img/<path:filePath>")
def getImage(filePath):
    fileName = os.path.basename(filePath)
    fileDirectory = filePath.split(fileName)[0]

    return send_from_directory(fileDirectory, fileName)


"""
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = 1.0 - face_match_threshold
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
"""
