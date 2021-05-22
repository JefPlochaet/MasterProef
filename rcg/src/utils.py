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

def get_input_seq(args, torchbatch, index):
    """Geeft de input sequentie van de batch"""

    return torchbatch[:, index:args.input_length+index]

def get_gt_frame(args, torchbatch, index):
    """Neemt de gt frame en zet deze om in een torch tensor"""

    return torchbatch[:, args.input_length+index: args.input_length+index+1]

def get_gt_seq(args, torchbatch, index):
    """Zal de input sequentie + 1 frame terug geven
    als gt sequentie (ookal is de outputseq langer, 
    het netwerk zal steeds maar 1 beeld voorspellen)"""

    return torchbatch[:, index:args.input_length+index+1]

def add_front_frame(args, torchbatch, frame):
    """Vervangt eerste frame van inputseq (na clone)"""
    
    new_batch = torchbatch.clone()
    new_batch[:, 0:1] = frame

    return new_batch

def add_back_frame(args, inseq, frame):
    """Voegt voorspelde frame aan inputsequentie toe"""
    
    new_batch = torch.cat([inseq, frame], dim=1)

    return new_batch