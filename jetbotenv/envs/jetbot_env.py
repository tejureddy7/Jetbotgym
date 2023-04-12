'''
This examples uses the Flow and Multi-ranger decks to move crazyflie in all directions
'''
import gym
import numpy as np
import sys
import copy
import random
import math
from gym import spaces, error
from enum import IntEnum
import logging
import sys
import torch
import cv2
import time
# from jetbotmini import Robot, Camera, bgr8_to_jpeg, Heartbeat
# import time
# from datetime import datetime
# import os
# import ipywidgets.widgets as widgets
# from jetbotmini import Robot, Camera, bgr8_to_jpeg, Heartbeat
# import traitlets
# import PIL
# from PIL import Image

class JetbotBaseEnv(gym.Env):

    # lets say we have 5 actions (hover, left, right, forward, back)
    class Actions(IntEnum):
        turn_left = 0
        turn_right = 1
        move_forward = 2

    def __init__(self, seed=None):

        # Action enumeration for this environment
        self.actions = JetbotBaseEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        #observation space for single object
        self.observation_space = gym.spaces.Box(low=-100000, high=100000, shape=(1*4,))
        obs_shape = self.observation_space.shape

        #load object detection model - yolo5n
        self.model_yolo = torch.hub.load('yolov5','custom', path='yolov5/Lastweightscolored.pt', source='local')

        #connect to Jetbot and create a camera instance
        #self.robot = Robot()
        #self.camera = Camera.instance(width = 80, height = 60)
        # image = widgets.Image(format='jpeg', width = 270, height = 270)
        #
        # #link to the browser and convert to jpeg for viewing
        # camera_link = traitlets.dlink((camera, 'value'), (image, 'value'), transform=bgr8_to_jpeg)
        # display(image)
        self.i = 0

    def get_img(self):
        #img = self.camera.value
        #img = img[:,:,::-1]
        img = cv2.imread('j2.PNG')
        #img = cv2.imread('3.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('{0:05d}.jpg'.format(i),img)
        return img

    def state(self, observation):
        obs = observation
        contrast = 1# Contrast control ( 0 to 127)
        brightness = 100 # Brightness control (0-100)

        # call addWeighted function. use beta = 0 to effectively only
        #operate on one image
        #out = cv2.addWeighted(obs, contrast, obs, 0, brightness)


        #obs = cv2.normalize(obs, None, alpha=50, beta=500, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # cv2.imshow("img",obs)
        # cv2.waitKey(0)
        # obs = cv2.cvtColor(observation, cv2.COLOR_BGRA2RGB)
        # obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)


        results = self.model_yolo(obs)
        BBox_Coordinates = results.pandas().xyxy[0].sort_values('xmin')
        x1,y1,x2,y2 = 0,0,0,0

        if(len(BBox_Coordinates)!=0):
            if(0 in BBox_Coordinates['class'].values): #blue box
                n1 = BBox_Coordinates.index[BBox_Coordinates['class'] == 0].values
                n1 = n1[0]
                x1 = int(BBox_Coordinates['xmin'][n1])
                y1 = int(BBox_Coordinates['ymin'][n1])
                x2 = int(BBox_Coordinates['xmax'][n1])
                y2 = int(BBox_Coordinates['ymax'][n1])
                print("Cube detected")
                new = cv2.rectangle(obs, (x1,y1),(x2,y2), (255, 255, 255), 2)
                cv2.imwrite('re.jpg',new)
            BBox = [x1,y1,x2,y2]



        else:
            BBox = [0,0,0,0]
            print("No Detection happened")

        obs = BBox
        obs = torch.FloatTensor([obs])
        cv2.destroyAllWindows()
        return  obs

    def reset(self):

        #self.robot.stop()
        obs = [0,0,0,0]
        obs = torch.FloatTensor([obs])
        #print(obs,obs.type)
        return obs

    def mov_fwd(self):
        print('moving forward')
        #self.robot.forward(0.15) #0.15 in miniworld
        time.sleep(0.5)
        #stop()

    def mov_lft(self):
        print('moving left')
        #self.robot.left(0.45) #15 degrees in miniworld
        time.sleep(0.2)
        #stop()

    def mov_rght(self):
        print('moving right')
        #self.robot.right(0.45) #15 degrees in miniworld
        time.sleep(0.2)
        #stop()

    def step(self, action):
        done = False
        reward = 0
        observation = self.get_img()
        obs = self.state(observation)

        if action == self.actions.turn_left:
            self.mov_lft()

        if action == self.actions.turn_right:
            self.mov_rght()

        if action == self.actions.move_forward:
            self.mov_fwd()

        return obs, reward, done, {}
