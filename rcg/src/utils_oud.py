import numpy as np
import torch

def convert_to_tensor(args, batch):
    """Zet de sequentie om in een torch tensor"""

    new_batch = []

    for seq in batch:
        new_seq = np.concatenate(seq[:], axis=0)

        new_batch.append(new_seq)
    
    arr = np.array(new_batch, dtype=np.float32)

    return torch.tensor(arr)

def get_input_seq(args, torchbatch):
    """Geeft de input sequentie van de batch"""

    return torchbatch[:, 0:args.input_length]

def get_gt_frame(args, torchbatch):
    """Neemt de gt frame en zet deze om in een torch tensor"""

    return torchbatch[:, args.input_length: args.input_length+1]

def get_gt_seq(args, torchbatch):
    """Zal de input sequentie + 1 frame terug geven
    als gt sequentie (ookal is de outputseq langer, 
    het netwerk zal steeds maar 1 beeld voorspellen)"""

    return torchbatch[:, 0:args.input_length+1]

def add_front_frame(args, torchbatch, frame):
    """Neemt gt sequence binnen en vervangt EERSTE frame
    met een voorspelde frame"""
    
    new_batch = torchbatch.clone()
    new_batch[:, 0:1] = frame

    return new_batch

def add_back_frame(args, torchbatch, frame):
    """Neemt gt sequence binnen en vervangt LAATSTE frame
    metmet een voorspelde frame"""
    
    new_batch = torchbatch.clone()
    new_batch[:, args.input_length:args.input_length+1] = frame

    return new_batch