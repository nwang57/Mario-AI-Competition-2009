import math
import numpy as np

class SegTree:

    '''
        | 0.1 | 0.4 | 0.05 | 0.01 | ... | 0.2 |        <- weights
                       |
                       +---- last inserted (end of queue), at i-th slot

        | 47  |  48 |  49  |   0  | ... | 46  |        <- indices
                 |
                 +---- calculated as [j-i + (n-1)]%n for slot j, n the total number of slots

        SegTree represented as an array
        +----+----+----+----+----+----+----+----+--------+----+----+----+--------
        | /  | L1 | L2 | L2 | L3 | L3 | L3 | L3 |  ....  |Ln-1| Ln | Ln |  ....     
        +----+----+----+----+----+----+----+----+--------+----+----+----+--------
           0    1    2    3    4    5    6    7    ....        2^(n-1)
    '''

    def __init__(self, size):
        self.num_slots = size
        self.num_layers = int(math.ceil(math.log(size+1)/math.log(2)))
        self.array_size = int(pow(2, self.num_layers))
        self.tree_size = 2 * self.array_size
        self.tree = [0.0] * self.tree_size
        self.last_slot = -1
        self.num_items = 0

    def get_idx(self, x):
        total_weight = self.tree[1]
        if x > total_weight:
            print('Error: weight selector negative.')
            return -1

        slot_id = 1
        while slot_id < self.array_size:
            left_child = self.tree[2 * slot_id]
            if x <= left_child:
                slot_id = slot_id * 2
            else:
                x -= left_child
                slot_id = slot_id * 2 + 1

        return slot_id - self.array_size

    def sample(self, batch_size):
        total_weight = self.tree[1]
        rand_select = np.random.uniform(low=0.0, high=total_weight, size=(batch_size,))
        return [self.get_idx(x) for x in rand_select]

    def change_weight(self, idx, new_weight):
        if idx >= self.num_slots:
            print('Error: idx out of range.')
            return

        pos_in_tree = self.array_size + \
            (idx + self.last_slot + 1 - self.num_items + self.num_slots) % self.num_slots
        old_weight = self.tree[pos_in_tree]
        while pos_in_tree > 0:
            self.tree[pos_in_tree] = self.tree[pos_in_tree] + new_weight - old_weight
            pos_in_tree //= 2

    def add(self, weight):
        if self.num_items >= self.num_slots:
            self.change_weight(0, weight) 
        else:
            self.change_weight(self.num_items, weight)
            self.num_items += 1

        self.last_slot = (self.last_slot + 1) % self.num_slots

    def get_max_weight(self):
        if self.num_items is 0:
            return 1.0
        return np.max(self.tree[self.array_size:])

    def get_total_weight(self):
        return self.tree[1]

    def _print_tree(self):
        accu = 0
        layer = 0
        for i in range(self.tree_size - 1):
            print(self.tree[i+1]),
            print(' '),
            if i - accu + 1 == pow(2, layer):
                layer += 1
                accu = i + 1
                print
