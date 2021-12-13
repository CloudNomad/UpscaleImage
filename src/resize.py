"""
Summary: Upscales images by factor of 4 and maintains aspect ratio
Credit: Saafke (github account) for trained models
"""

# Will add noise removal

import os

dirname = os.path.dirname(__file__)
imagePath = os.path.join(dirname, '../imgs/CrashBandicoot1996.jpg')
imgDst = os.path.join(dirname, '../dst/')

import cv2
from cv2 import dnn_superres

# Use deep neural network to generate super res object for model
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread(imagePath)

# Use fast super-resolution convolutional neural network (FSRCNN) and scale by factor of 4
path = "FSRCNN-small_x4.pb"
sr.readModel(path)
sr.setModel("fsrcnn", 4)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite(imgDst + 'rescaled_img.png', result)
