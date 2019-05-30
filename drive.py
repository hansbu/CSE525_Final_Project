#parsing command line arguments
import argparse
#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking
import eventlet
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO
#load our saved model
import torch
from torch.autograd import Variable
from models import *
import torchvision.transforms as transforms
#helper class
import utils_autoCar as utils
import torchvision.transforms as T
import time

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#init our model and image array as empty
model = None
prev_image_array = None

#set min/max speed for our autonomous car
MAX_SPEED = 25
MIN_SPEED = 10

#and a speed limit
speed_limit = MAX_SPEED

# transformations = transforms.Compose([transforms.Lambda(lambda x: x/127.5 - 1)])

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral
controller = SimplePIController(0.1, 0.002)   # origin 0.1, 0.002
set_speed = 25
controller.set_desired(set_speed)
count = 0

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])

        # The current image from the center camera of the car
        original_image = Image.open(BytesIO(base64.b64decode(data["image"])))
        # image = utils.readImg(BytesIO(base64.b64decode(data["image"])))

        try:
            # original_image.save('auto.png')
            image = np.asarray(original_image)       # from PIL image to numpy array

#            image = utils.preprocess(image) # apply the preprocessing
            # image = transformations(image)
            start = time.time()

            image = utils.preprocesses(image)
            #
            transform  = T.Compose([T.Lambda(lambda x: x/127.5 - 1)])
            # transform = T.ToTensor()
            image = transform(image)

            image = torch.FloatTensor(image)
            image = image.permute(2,0,1).contiguous()       # swap the axes

            image = image.view(1, 3, 66, 200)

            image = Variable(image)

            # predict the steering angle for the image
            steering_angle = float(model(image).view(-1).data.numpy()[0])

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED

            if speed < 0.1: speed = MIN_SPEED*np.random.uniform()

            steering_angle = steering_angle * 2.1
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            # throttle = controller.update(float(speed)) - 0.1
            # throttle = controller.update(float(speed))
            # throttle = max(throttle, -0.15 / 0.05 * abs(steering_angle) + 0.35)
            global count
            count += 1
            if (count > 100 and speed < 0.1):
                while(speed < 0.1):
     #               print('Software Bug!')
                    count = 101
                    throttle = 0.5*np.random.uniform()
                    send_control(steering_angle, throttle)
            else:
               send_control(steering_angle, throttle)

            stop = time.time() - start
            noFrame = int(1 / stop)
            # print('Number of frame per second: ', noFrame)
            print('%.4f %.4f %.2f %d' % (steering_angle, throttle, speed, noFrame))


        except Exception as e:
            print("Exception")
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            original_image.save('{}.jpg'.format(image_filename))
    else:

        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = model_car_3()
#    model.eval()
    utils.load_net(args.model, model)



    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
