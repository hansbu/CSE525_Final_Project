from __future__ import print_function, division
import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np

# define model
class model_car_1(nn.Module):
    def __init__(self):
        super(model_car_1, self).__init__()
        # input is 80x320

        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(32, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=(4, 4), stride=4),
        )

        self.linears = nn.Sequential(
            nn.Linear(240, 128),
            nn.ELU(),
            nn.Linear(128, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
        )

        self._initialize_weights()

    def forward(self, input):
        output = self.convs(input)
        output = output.view(output.size(0), -1)  # flatten output
        output = self.linears(output)
        return output

    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)



class model_car_2(nn.Module):
    def __init__(self):
        super(model_car_2, self).__init__()
        # input is 66x200

        self.convs = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.BatchNorm2d(36),

            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.BatchNorm2d(64),

        )

        self.linears = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
        )

        self._initialize_weights()

    def forward(self, input):
        output = self.convs(input)
        output = output.view(output.size(0), -1)  # flatten output
        output = self.linears(output)
        return output

    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)




class model_car_3(nn.Module):
    def __init__(self):
        super(model_car_3, self).__init__()
        # input is 66x200
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.linears = nn.Sequential(
            nn.Dropout2d(0.25),
            nn.Linear(3648, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        self._initialize_weights()

    def forward(self, input):
        output = self.convs(input)
        output = output.view(output.size(0), -1)  # flatten output
        output = self.linears(output)
        return output

    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)






class model_car_4(nn.Module):
    def __init__(self):
        super(model_car_4, self).__init__()
        # input is 66x200

        self.convs = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Dropout2d(),

        )

        self.linears = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
        )

        self._initialize_weights()

    def forward(self, input):
        output = self.convs(input)
        output = output.view(output.size(0), -1)  # flatten output
        output = self.linears(output)
        return output

    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)

class model_car_5(nn.Module):
    def __init__(self):
        super(model_car_5, self).__init__()
        # input is 66x200
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.linears = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(3648, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        self._initialize_weights()

    def forward(self, input):
        output = self.convs(input)
        output = output.view(output.size(0), -1)  # flatten output
        output = self.linears(output)
        return output

    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)





class model_car_6(nn.Module):
    def __init__(self):
        super(model_car_6, self).__init__()
        # input is 66x200
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.linears = nn.Sequential(
            nn.Dropout2d(0.25),
            nn.Linear(2176, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        self._initialize_weights()

    def forward(self, input):
        output = self.convs(input)
        output = output.view(output.size(0), -1)  # flatten output
        output = self.linears(output)
        return output

    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)



