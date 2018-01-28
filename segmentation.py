#from image_import import import_images
from image_import_2 import load_images
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

'''
#paths of images and masks
mask_path=("/home/parth/Downloads/VOCdevkit/VOC2012/SegmentationClass/*.png")
image_path=("/home/parth/Downloads/VOCdevkit/VOC2012/JPEGImages/*.jpg")
tr_image_save_path = ("/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/segmentation_train/")
tr_mask_save_path = ("/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/segmentation_train_masks/")
test_image_save_path = ("/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/segmentation_test/")
test_mask_save_path = ("/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/segmentation_test_masks/")


#lists containing split-up of training and test images
train_list=pd.read_csv("/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/Segmentation/train.csv")
train_list=np.array(train_list)
test_list=pd.read_csv("/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/Segmentation/val.csv")
test_list=np.array(test_list)

#training_images = import_images(image_path, tr_image_save_path, train_list)
#training_masks = import_images(mask_path, tr_mask_save_path, train_list)

#test_images = import_images(image_path, test_image_save_path, test_list)
test_masks = import_images(mask_path, test_mask_save_path, test_list)

#print(np.shape(training_images))
#print(len(training_masks))

#print(len(test_images))
print(len(test_masks))
'''

train_image_path=('/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/segmentation_train/*.png')
train_mask_path=('/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/segmentation_train_masks/*.png')
train_list=pd.read_csv("/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/Segmentation/train.csv")

test_image_path=('/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/segmentation_test/*.png')
test_mask_path=('/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/segmentation_test_masks/*.png')
test_list=pd.read_csv("/home/parth/Downloads/VOCdevkit/VOC2012/ImageSets/Segmentation/val.csv")

def load_batch(images):
    batch = []
    for i in range(32):
        k = np.random.randint(1436)
        batch.append(images[k])
    return (batch)

img_rows = 496
img_cols = 496

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():

    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

model=get_unet()
training_images = load_images(train_image_path, train_list)
training_masks = load_images(train_mask_path, train_list)

test_images = load_images(test_image_path, test_list)
test_masks = load_images(test_mask_path, test_list)

for i in range(45):
    training_batch = load_batch(training_images)
    training_masks_batch = load_batch(training_masks)

    test_batch = load_batch(test_images)
    test_masks_batch = load_batch(test_masks)

    model.train_on_batch(np.array(training_batch), np.array(training_masks_batch))
    training_loss=model.evaluate(np.array(training_batch), np.array(training_masks_batch))
    test_loss=model.test_on_batch(np.array(test_batch), np.array(test_masks_batch))

    print(training_loss, test_loss)