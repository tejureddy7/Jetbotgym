import ipywidgets.widgets as widgets
from jetbotmini import Robot, Camera, bgr8_to_jpeg, Heartbeat
import traitlets
import time
from datetime import datetime
import PIL
from PIL import Image
import os

'''
    0: Forward
    1: Left
    2: Right
'''

def stop(change):
    robot.stop()
    cntr += 1

def mov_fwd(change):
    img = camera.value
    img = Image.fromarray(img[:,:,::-1])
    img.save(epi_path[0] + str(cntr[0]) + ".jpg")
    robot_actions.append(0)
    robot.forward(0.4)
    time.sleep(0.5)
    robot.stop()
    cntr[0] += 1

def mov_bckwd(change):
    robot.backward(0.45)
    time.sleep(0.1)

def mov_lft(change):
    img = camera.value
    img = Image.fromarray(img[:,:,::-1])
    img.save(epi_path[0] + str(cntr[0]) + ".jpg")
    robot_actions.append(1)
    robot.left(0.45)
    time.sleep(0.2)
    robot.stop()
    cntr[0] += 1

def mov_rght(change):
    img = camera.value
    img = Image.fromarray(img[:,:,::-1])
    img.save(epi_path[0] + str(cntr[0]) + ".jpg")
    robot_actions.append(2)
    robot.right(0.45)
    time.sleep(0.2)
    robot.stop()
    cntr[0] += 1

    

def main():

    global robot, heartbeat, flag, image, cntr, epi_path, camera, robot_actions

    flag = 0

    robot = Robot()

    ts = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

#     create a camera instance and display it
    camera = Camera.instance(width = 224, height = 224)
    image = widgets.Image(format='jpeg', width = 270, height = 270)
#     link to the browser and convert to jpeg for viewing
    camera_link = traitlets.dlink((camera, 'value'), (image, 'value'), transform=bgr8_to_jpeg)
    display(image)

    robot_actions = []

    epi_path = []
    cntr = []
    epi_path.append('./data/episode_'+str(ts)+'/')
    cntr.append(int(1))

    os.mkdir(epi_path[0])

    #create a virtual controller and link their actions to drive the bot
    make_controller()

    #stop the bot if we lose connection to it
    heartbeat = Heartbeat()

    heartbeat.observe(handle_heartbeat_status, names='status')

    period_slider = widgets.FloatSlider(description='period', min=0.001, max=0.5, step=0.01, value=0.5)
    traitlets.dlink((period_slider, 'value'), (heartbeat, 'period'))

    display(period_slider, heartbeat.pulseout)
