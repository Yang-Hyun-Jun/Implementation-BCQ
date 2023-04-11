import numpy as np
import gymnasium as gym
import torch
import os

from bcq import BCQ

def tensorize(array):
    array = np.array(array).reshape(1, -1)
    return torch.tensor(array).float()

if __name__ == '__main__':
    
    env = gym.make('Hopper-v4', render_mode='human')
    
    param = {
        's_dim':11, 
        'a_dim':3,
        'gamma':0.99, 
        'tau':0.005, 
        'lam':0.75
        }

    bcq = BCQ(**param)
    bcq.vae.load_state_dict(torch.load(os.getcwd() + '/Metrics/vae.pth'))
    bcq.perturb.load_state_dict(torch.load(os.getcwd() + '/Metrics/perturb.pth'))
    bcq.qnet.load_state_dict(torch.load(os.getcwd()+ '/Metrics/qnet.pth'))

    bcq.vae.eval()
    bcq.perturb.eval()
    bcq.qnet.eval()
    
    for epi in range(1, 100 + 1):
        state, _ = env.reset()
        env.render()
        cumr = 0

        while True:
            action = bcq.get_action(tensorize(state))
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            cumr += reward

            if done:
                print(f'cumr:{cumr}')
                break




