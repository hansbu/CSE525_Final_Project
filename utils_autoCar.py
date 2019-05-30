from __future__ import print_function, division
import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import cv2

width, height = 200, 66
# width, height = 220, 60

# usage: save_net('modelname.h', NET)
def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

# Load model .h
def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def Write2Txt(fileName, contents, mode='a'):
    with open(fileName, mode) as text_file:
        text_file.write(contents)
        text_file.write('\n')

def resize(image):
    return cv2.resize(image, (width, height), cv2.INTER_AREA)

def dataset_placeholder(conf, val_size = 0.3, test_size = 0.0):
    # split all dataset into train, val, test
    count = 0
    for imgDir in conf.imgDirs:
        csv_fileName = conf.parentDir + '/' + imgDir + '.csv'
        data = pd.read_csv(csv_fileName,
                           names=['img_center', 'img_left', 'img_right', 'steer', 'throttle', 'reverse', 'speed'])
        X_temp = data[['img_center', 'img_left', 'img_right']].values
        Y_temp = data['steer'].values.reshape(-1, 1)

        # currently the path of X is not correct, we need to get the image name only then concatinate with parent folder
        for i in range(np.shape(X_temp)[0]):
            for j in range(np.shape(X_temp)[1]):
                temp = X_temp[i][j].split('/')[-1]
                X_temp[i][j] = os.path.join(conf.parentDir, imgDir, temp)

        if count == 0:
            X = X_temp.copy()
            Y = Y_temp.copy()
        else:
            X = np.concatenate((X, X_temp), axis=0)
            Y = np.concatenate((Y, Y_temp), axis=0)
        count = 1

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size, random_state=0, shuffle=True)
    # X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=test_size, random_state=0, shuffle=True)
    return X_train, Y_train, X_val, Y_val

def readImg(path):
    # output is a numpy array
    img = Image.open(path)
    img = np.asarray(img)
    return img

def rgb2YUV(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img_yuv

def rgb2HSV(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img_hsv

def crop_img(img):
    img = img[60:-25,:,:]
    return img

def preprocesses(img):
    img = crop_img(img)     # crop image
    img = resize(img)
    # img = rgb2HSV(img)    # other option: convert to hsv
    img = rgb2YUV(img)      # convert to YUV
    return img

def randomFlipImg(img, steering_angle = 0.1):
# randomly flip the img and its corresponding steering angle
    if np.random.rand() > 0.5:      # 50% to flip
        img = cv2.flip( img, 1)      # flip left to right
        steering_angle = -steering_angle
    return img, steering_angle

def randomTranslate(img, steering_angle, delta_x = 100, delta_y = 10):
    # usage: randomTranslate(img, steering_angle, 100, 10)
    trans_x = delta_x * (np.random.rand() - 0.5)        # random from -0.5 to 0.5
    trans_y = delta_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    # print('trans_x: %.2f, steering angle shift %.2f, steering angle: %.2f' % (trans_x, trans_x * 0.002, steering_angle))
    M = np.float32([[1, 0, trans_x], [0, 1, trans_y]]) # affine matrix, no rotation
    h, w = img.shape[:2]
    img = cv2.warpAffine(img, M, (w, h))
    img = np.asarray(img, dtype='uint8')
    return img, steering_angle

def randomBrightness(img):
    # HSV: V for value of brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)      # convert to HSV
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)    # random -0.5 to 0.5
    hsv[:,:,2] =  hsv[:,:,2] * ratio                # adjust the brightness by changing the V channel
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augmentation(img, steering_angle):
    # do augmentation
    img, steering_angle = randomFlipImg(img, steering_angle=steering_angle)
    img, steering_angle = randomTranslate(img, steering_angle)
    img = randomBrightness(img)
    return img, steering_angle

def normalizeData(img):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(img)

class Dataset_cars(Dataset):

    def __init__(self, X, Y, is_training, transform=None):
        self.X = X
        self.Y = Y
        self.is_training = is_training
        self.transform = transform

    def __getitem__(self, index):
        center, left, right = self.X[index]
        steering_angle = self.Y[index]
        i = np.random.randint(0,3)
        img = 0.0
        angle = 0.0

        if self.is_training and np.random.random() < 0.6:   # if training and augmentation
            # perform augmentation
            if i == 1:  # left
                img = readImg(left)
                angle = steering_angle + 0.1
                img, angle = augmentation(img, angle)
            elif i == 2:    # right
                img = readImg(right)
                angle = steering_angle - 0.1
                img, angle = augmentation(img, angle)
            else:
                img = readImg(center)
                img, angle = augmentation(img, steering_angle)

        else:
            img = readImg(center)
            angle = steering_angle.copy()

        img = preprocesses(img)

        if self.transform is not None:
            img = self.transform(img)

            img = torch.FloatTensor(img)
            img = img.permute(2,0,1).contiguous()       # swap the axes

        return img, angle

    def __len__(self):
        return self.X.shape[0]

def to_var(x, dtype=torch.FloatTensor):
    x = x.type(dtype)  # Construct a PyTorch Variable out of your input data
    if torch.cuda.is_available():
        return Variable(x.cuda())
    return Variable(x)
