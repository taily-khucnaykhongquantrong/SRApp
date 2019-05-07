import matlab.engine
from PIL import Image

eng = matlab.engine.start_matlab()
img = Image.open('imagePath')
img_mat = matlab.uint8(list(img.getdata()))
img_mat.reshape((img.size[0], img.size[1], 3))

score = eng.quality_predict(img_mat)
