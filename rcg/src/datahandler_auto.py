import numpy as np
import random

class DataHandler():
    def __init__(self, args, path):
        self.args = args
        self.path = path
        self.data = []

        self.current_position = 0
        self.num_sequences = 0
        self.batch_size = args.batch_size
        self.current_batch_size = 0

        self.load() 

    def load(self):
        self.data = np.load(self.path)

        self.num_sequences = len(self.data)


    def begin(self, shuffle=True):
        if shuffle:
            random.shuffle(self.data)
        self.current_position = 0

        if self.current_position + self.batch_size > self.num_sequences:
            self.current_batch_size = self.num_sequences - self.current_position
        else:
            self.current_batch_size = self.batch_size

    def next(self):
        self.current_position += self.current_batch_size
        if not self.batch_left():
            return None
        
        if self.current_position + self.batch_size > self.num_sequences:
            self.current_batch_size = self.num_sequences - self.current_position
        else:
            self.current_batch_size = self.batch_size


    def batch_left(self):
        if self.current_batch_size < self.num_sequences - self.current_position:
            return True
        else:
            return False 
    
    def get_batch(self):
        batch = self.data[self.current_position:self.current_position + self.current_batch_size]

        return batch
    
    def get_revbatch(self):
        rev_batch = []

        batch = self.get_batch()

        for sequence in batch:
            newsequence = sequence[::-1, :, :, :]
            
            rev_batch.append(newsequence)
        

        return np.array(rev_batch)
        