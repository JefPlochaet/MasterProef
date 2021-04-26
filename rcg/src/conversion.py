import numpy as np
import torch


def convert_inputseq(args, batch):
    """Zet de input sequentie om in een torch tensor"""

    new_batch = []

    for seq in batch:
        new_seq = np.concatenate(seq[0:args.input_length], axis=0)

        new_batch.append(new_seq)
    
    arr = np.array(new_batch, dtype=np.float32)

    return torch.tensor(arr)

def get_gt_frame(args, batch):
    """Neemt de gt frame en zet deze om in een torch tensor"""

    new_batch = []

    for seq in batch:
        new_batch.append(seq[args.input_length])

    arr = np.array(new_batch, dtype=np.float32)

    return torch.tensor(arr)

def convert_totalseq(args, batch):
    """Zet volledige sequentie om in een torch tensor"""

    new_batch = []

    for seq in batch:
        new_seq = np.concatenate(seq[:], axis=0)

        new_batch.append(new_seq)

    arr = np.array(new_batch, dtype=np.float32)

    return torch.tensor(new_batch)

def get_gt_seq(args, batch):
    """Zal de input sequentie + 1 frame terug geven
    als gt sequentie (ookal is de outputseq langer, 
    het netwerk zal steeds maar 1 beeld voorspellen)"""

    new_batch = []

    for seq in batch:
        new_seq = np.concatenate(seq[0:args.input_length+1], axis=0)

        new_batch.append(new_seq)

    arr = np.array(new_batch, dtype=np.float32)

    print(arr.shape)

    return torch.tensor(new_batch, dtype=torch.float)

def add_predframe_to_seq(args, seq, predframe):
    """Plakt de voorspelde frame aan de sequentie vooraan
    (om de accent accent frames te voorspellen)"""
    
    for i in range(len(seq)):
        for ch in range(args.img_ch):

            seq[i][ch] = predframe[i][ch]

    return seq

def make_new_seq(args, gtseq, predframe):
    """Plakt voorspelde frame achteraan de sequentie"""

    for i in range(len(gtseq)) :
        for ch in range(args.img_ch):

            gtseq[i][(args.input_length * args.img_ch) + ch] = predframe[i][ch]
    
    return gtseq
    

