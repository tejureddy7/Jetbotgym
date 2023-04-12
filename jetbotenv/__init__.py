from gym.envs.registration import register

register(
    id='JetbotBaseEnv-v0',
    entry_point='jetbotenv.envs:JetbotBaseEnv'
)
