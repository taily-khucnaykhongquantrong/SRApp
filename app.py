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

import test
import utils.util as util

UPLOAD_FOLDER = "img/LR/"
RESULT_FOLDER = "img/sr/"
GIF_FOLDER = "img/gif/*"
SECRET_KEY = b'_5#y2L"F4Q8z\n\xec]/'

app = Flask(__name__)
# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = SECRET_KEY

util.mkdir(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


@app.route("/")
def index():
    # Extract only webm file type
    gifImages = [path for path in sorted(glob(GIF_FOLDER)) if path.endswith(".webm")]
    return render_template("index.html", gifImages=gifImages)


@app.route("/video/<path:filePath>")
def getVideo(filePath):
    return send_from_directory(
        filePath.split(os.path.basename(filePath))[0], os.path.basename(filePath)
    )


@app.route("/result", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        # Get the name of the uploaded files
        uploaded_files = request.files.getlist("file[]")
        filenames = []

        for file in uploaded_files:

            # Check if the file is one of the allowed types/extensions
            if file and util.allowed_file(file.filename):
                # Make the filename safe, remove unsupported chars
                filename = secure_filename(file.filename)
                # Move the file form the temporal folder to the upload
                # folder we setup
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                # Save the filename into a list, we'll use it later
                filenames.append(filename)
                # Redirect the user to the uploaded_file route, which
                # will basicaly show on the browser the uploaded file
        # Load an html page with a link to each uploaded file

        test.main("./options/test/test_ESRGAN.json")

        lrFiles = [file for file in os.listdir(UPLOAD_FOLDER)]
        hrFiles = [file for file in os.listdir(RESULT_FOLDER)]

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

        return render_template("result.html", lrFiles=lrFiles, hrFiles=hrFiles)


@app.route("/hr/<path:filename>")
def lr(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/sr/<path:filename>")
def hr(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)


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
