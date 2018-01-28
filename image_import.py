import numpy as np
import glob
import cv2
import pandas as pd
import os


#import images from given path using glob for navigation(some extra work for VOC2012)
def import_images(path , save_path, image_list):

    images = []
    for i in range(len(image_list)):
        for img in glob.glob(path):
            name, extension = str(os.path.split(img)[-1]).split(".")
            if name == image_list[i]:
                image = cv2.imread(img)
                image = cv2.resize(image, (496, 496))
                cv2.imwrite(save_path + name + ".png", image)
                images.append(image)

    return images









