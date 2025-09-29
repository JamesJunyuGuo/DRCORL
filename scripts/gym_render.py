import matplotlib.pyplot as plt
# from utils.utils import *
import gymnasium as gym

env_list = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Swimmer-v4", "Walker2d-v4"]

for task in env_list:
    env = gym.make(task, render_mode="rgb_array")
    env.reset()
    img = env.render()

    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f"{task}.png", bbox_inches='tight')

    env.close()



    

    
    