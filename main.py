import random 
import torch
from bcq import BCQ

class RelayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []

    def push(self, data):
        self.memory.append(data)
        if len(self.memory) > self.buffer_size:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

if __name__ == "__main__":
        
    buffer = torch.load('Metrics/buffer.pth')
    
    param = {
        's_dim':11, 
        'a_dim':3,
        'gamma':0.99, 
        'tau':0.005, 
        'lam':0.75
        }
    
    steps = 400000
    batch_size =100

    bcq = BCQ(**param)
    bcq.train(buffer, steps=steps, batch_size=100)

    
    

