import random
import torch

class Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []

    def push(self, data):
        # memory에 overflow나면 oldest부터 교체
        self.memory.append(data)
        if len(self.memory) > self.buffer_size:
            del self.memory[0]

    def sample(self, batch_size):
        # random하게 batch_size만큼 sampling 함
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
off_buffer = torch.load('Metrics/buffer.pth')

