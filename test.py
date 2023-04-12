import gym
import jetbotenv
env = gym.make('JetbotBaseEnv-v0')

for episode in range(10):
    obs = env.reset()
    for step in range(50):
        action = env.action_space.sample() # take a random action
        env.step(action)
        state, reward, done, _ = env.step(action)

    #see bounding box on images
    #cv2.imshow("ResultOutputWindow",obs)
    #cv2.waitKey(1)

    # cv2.imshow("OutputWindow",observation)
    # cv2.waitKey(1) #blue

    #cv2.imwrite('C:/Users/EEHPC/Airlearning_project2/airlearning-rl2/yoloimagestest/1.png',rgb)
    #results = self.model_yolo('C:/Users/EEHPC/Airlearning_project2/airlearning-rl2/yoloimagestest/1.png')
    #cv2.rectangle(obs, (x1, y1), (x2, y2), (255, 255, 255), 2)
