"""
Summary: Upscales images by factor of 4 and maintains aspect ratio
Credit: Saafke (github account) for trained models
"""

# Will add noise removal

import os
import cv2
from cv2 import dnn_superres
from PIL import Image
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

    print('\nImage rescaling complete!')

except Exception:
    print('Image rescaling failed :(')