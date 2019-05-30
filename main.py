from __future__ import print_function, division
import argparse
import pandas as pd
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader,sampler,Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as opt
import time
from models import *
from utils_autoCar import *

# import matplotlib.pyplot as plt

conf = {'parentDir':'data',
        'imgDirs':[] ,'batch_size':16, 'No_epoch':200,
        'learningRate':0.0001, 'save_step':100,
        'logfile':'log_train.txt', 'modelName':''}

conf = argparse.Namespace(**conf)

dirs = ['800_600_simple_2_1', '800_600_simple_2_2', '800_600_simple_2_3',
        '800_600_simple_2_4', '800_600_simple_2_5',
        '800_600_fastest_2_1', '800_600_fastest_2_2', '800_600_fastest_2_3',
        '800_600_fastest_2_4', '800_600_fastest_2_5',
        '800_600_beautiful_2_1', '800_600_beautiful_2_2', '800_600_beautiful_2_3',
        '800_600_beautiful_2_4', '800_600_beautiful_2_5'
        ]


conf.imgDirs = dirs
specificRun = str(conf.learningRate) + '_Adam_Road2'

conf.logfile = 'exp512_4_model3_norm01_' + '_lr_' + specificRun + '.txt'
conf.modelName = 'exp512_4_model3_norm01_' + 'lr_' + specificRun

# testing data02

if torch.cuda.is_available():
    conf.parentDir = os.path.join(os.getcwd(), 'data')

print('torch version: ', torch.__version__)
import platform
print('python version: ', platform.python_version())
print('data folder: ', conf.parentDir)

dtype = torch.FloatTensor # the CPU datatype

CUDA = torch.cuda.is_available()
seed = 123456
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)

X_train, Y_train, X_val, Y_val = dataset_placeholder(conf)


transform  = T.Compose([T.Lambda(lambda x: x/127.5 - 1)])
# transform = T.ToTensor()

train_set = Dataset_cars(X_train, Y_train, is_training=True, transform = transform)
train_loader = DataLoader(train_set, batch_size=conf.batch_size, shuffle=True, num_workers=4)

val_set = Dataset_cars(X_val, Y_val, is_training=False, transform = transform)
val_loader = DataLoader(val_set, batch_size=conf.batch_size, shuffle=False, num_workers=4)

# test_set = Dataset_cars(X_test, Y_test, is_training=False, transform = transform)
# test_loader = DataLoader(test_set, batch_size=conf.batch_size, shuffle=False, num_workers=4)

def train_model(net, optimizer, crit, train_data, val_data = None, No_epoch = 200):
    LOSS = []
    save_step = conf.save_step
    loss_sum = 0
    mae_best = 100000
    mse_best = 100000
    for epoch in range(No_epoch):
        net.train()
        loss_count = 0
        loss_sum = 0
        for i, data in enumerate(train_data):
            # center, left, right = data
            # views = [center, left, right]
            # for view in views:
            img, label = data
            img, label = to_var(img), to_var(label)
            output = net(img)
            loss = crit(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.data[0]
            loss_count += 1
    # if i % save_step == 0:
        loss_sum /= loss_count
        text_disp = 'Epoch: %d; Loss: %.4f; %s' % (epoch + 1, loss_sum, conf.modelName)
        print(text_disp)
        Write2Txt(conf.logfile, text_disp, mode = 'a')
        LOSS.append(loss_sum)

        if (epoch + 1) % 2 == 0 and val_data is not None:
            mae, mse = model_eval(net, val_data)
            if mse < mse_best:
                mse_best = mse
                save_net(conf.modelName + '_BEST_' + str(epoch + 1) + '_' + str(mse) + '.h5', net)
            else:
                save_net(conf.modelName + '_saved_' + str(epoch + 1) + '_' + str(mse) + '.h5', net)


            if mae < mae_best:
                mae_best = mae

            text_disp = 'Epoch: %d; MAE: %.4f; MAE_best: %.4f; MSE: %.4f; MSE_best: %.4f; %s' % (epoch + 1, mae, mae_best, mse, mse_best, conf.modelName)
            Write2Txt(conf.logfile, text_disp, mode = 'a')
            print(text_disp)
    return LOSS

def model_eval(net, data_loader):
    net.eval()
    if CUDA:
        net.cuda()
    mae = 0
    mse = 0
    for i, data in enumerate(data_loader):
        # center, left, right = data
        # views = [center, left, right]
        # for view in views:
        img, label = data
        # img = img.permute(0, 3, 1, 2).contiguous()

        img = to_var(img)
        output = net(img)
        output = output.data.cpu().numpy()
        label = label.numpy()

        mae += np.sum(abs(output-label))
        mse += np.sum(np.multiply((output-label), (output-label)))

    mae = mae/(len(data_loader)*conf.batch_size)
    mse = np.sqrt(mse/(len(data_loader)*conf.batch_size))
    return mae, mse

model = model_car_3()

input = torch.randn(16,3,66,200)
input = Variable(input)
output = model(input)
print('size of output: ', output.size())

if CUDA:
    model.cuda()
optimizer = opt.Adam(model.parameters(), lr = conf.learningRate)
criterion = nn.MSELoss().type(dtype)
if CUDA:
    criterion.cuda()

start = time.time()
if os.path.isfile(conf.logfile):
    os.remove(conf.logfile)

LOSS = train_model(model, optimizer, criterion, train_loader, val_loader, conf.No_epoch)
print('DONE, total time: ', time.time() - start)

# plt.plot([i for i in range(len(LOSS))], LOSS)
# plt.show()



