from collections import deque
import numpy as np
from segtree import SegTree

class Memory(object):
    def __init__(self, memory_size=10000, burn_in_size=5000):
        self.memory_size = memory_size
        self.burn_in_size = burn_in_size
        self.cur_size = 0
        self.memo = deque()

class classicMemory(Memory):
    def __init__(self, memory_size=10000, burn_in_size=5000):
        super(classicMemory, self).__init__(memory_size, burn_in_size)

    def append(self, transition):
        if self.cur_size < self.memory_size:
            self.cur_size += 1
        else:
            self.memo.popleft()
        self.memo.append(transition)

    def sample(self, batch_size=32):
        batch_index = np.random.randint(self.cur_size , size = batch_size)
        batch = []
        for idx in batch_index:
            batch.append(self.memo[idx])
        return np.array(batch), batch_index

    def get_total_weight(self):
        return 1

class prioritizedMemory(Memory):
    def __init__(self, memory_size=10000, burn_in_size=5000):
        super(prioritizedMemory, self).__init__(memory_size, burn_in_size)
        self.weightST = SegTree(memory_size)

    def append(self, transition, weight=None):
        if self.cur_size < self.memory_size:
            self.cur_size += 1
        else:
            self.memo.popleft()
        self.memo.append(transition)
        if weight is None:
            weight = self.weightST.get_max_weight()
        self.weightST.add(weight)

    def change_weight(self, idx, new_weight):
        self.weightST.change_weight(idx, new_weight)

    def get_total_weight(self):
        return self.weightST.get_total_weight()

    def sample(self, batch_size=32):
        batch_index = self.weightST.sample(batch_size)
        batch = []
        for idx in batch_index:
            batch.append(self.memo[idx])
        return np.array(batch), np.array(batch_index)
