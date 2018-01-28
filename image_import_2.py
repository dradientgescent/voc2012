import numpy as np
import glob
import cv2
import pandas as pd
import os
import tensorflow as tf

#import images from given path using glob for navigation(some extra work for VOC2012)
def load_images(path, list):

    images = []
    batch=[]


    for img in glob.glob(path):
        name, extension = str(os.path.split(img)[-1]).split(".")
        image = cv2.imread(img)
        images.append(image)
    return (images)


    #for i in range(32):
            #k=np.random.randint(1436)
            #batch.append(images[k])

    return(images)

