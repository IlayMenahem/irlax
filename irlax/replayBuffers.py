import collections
import random

class ReplayBuffer(object):
    """A simple Python replay buffer."""
    def __init__(self, capacity, batch_size):
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, data):
        self.buffer.append(data)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return tuple(zip(*batch))

    def is_ready(self):
        return len(self.buffer) >= self.batch_size
