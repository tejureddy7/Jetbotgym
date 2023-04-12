import torch
import sys
import gym
import jetbotenv
from torchsummary import summary
env = gym.make('JetbotBaseEnv-v0')

sys.path.append('../Jetbotgym/jetbotenv/miniworld_model')

actor_critic, ob_rms = \
            torch.load('jetbotenv/miniworld_model/trained_models/MiniWorld-Hallway-v0_yolo11_98.pt')
# print(actor_critic)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
device = torch.device("cpu")
model = actor_critic.to(device)

print(summary(model, (4,1)))

print("check",actor_critic.eval())


recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()


while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=True)
        print('Action',action)

    # observ reward and next obs
    obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)
