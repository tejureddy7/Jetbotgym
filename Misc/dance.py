from jetbotmini import Robot, Camera, bgr8_to_jpeg, Heartbeat
import time
from datetime import datetime
import os


def stop():
    robot.stop()

def mov_fwd():
    robot.forward(0.4)
    time.sleep(0.5)
    stop()

def mov_bckwd():
    robot.backward(0.4)
    time.sleep(0.5)
    stop()

def mov_lft():
    robot.left(0.45)
    time.sleep(0.2)
    stop()

def mov_rght():
    robot.right(0.45)
    time.sleep(0.2)
    stop()


robot = Robot()


for i in range(2):
    mov_lft()
    time.sleep(1)
    mov_rght()
    time.sleep(1)
    mov_fwd()
    time.sleep(1)
    mov_bckwd()
    time.sleep(1)
