"""
Summary: Upscales images by factor of 4 and maintains aspect ratio
Credit: Saafke (github account) for trained models
"""

# Will add noise removal

import os
import cv2
from cv2 import dnn_superres
from PIL import Image
import argparse
import time
import os
import numpy as np

dirname = os.path.dirname(__file__)
imagePath = os.path.join(dirname, '../imgs/CrashBandicoot1996.jpg')
imgDst = os.path.join(dirname, '../dst/')

try:
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
    imagedata_original = cv2.imread(imgDst + 'rescaled_img.png')
    cv2.waitKey(0)

    # Sharpening kernel
    sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    # Apply kernel to input image to get sharpened image
    sharpened_image = cv2.filter2D(imagedata_original, -1, sharpening_kernel)

    cv2.imwrite(imgDst + 'sharpened_img.png', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # set the base width of the result
    # basewidth = 3840
    # img = Image.open(imgDst + 'sharpened_img.png')
    # # determining the height ratio
    # wpercent = (basewidth/float(img.size[0]))
    # hsize = int((float(img.size[1])*float(wpercent)))
    # # resize image and save
    # img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    # img.save('new_image.jpg')

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
	    help="path to super resolution model")
    ap.add_argument("-i", "--image", required=True,
	    help="path to input image we want to increase resolution of")
    args = vars(ap.parse_args())

    # modelName = args["model"].split(os.path.sep)[-1].split("_")[0].lower()
    # modelScale = args["model"].split("_x")[-1]
    # modelScale = int(modelScale[:modelScale.find(".")])
    
    sr.readModel(args["model"])
    sr.setModel(modelName, modelScale)

    print('\nImage rescaling complete!')

except Exception:
    print('Image rescaling failed :(')