import argparse
import torch
import functools
# from torchinfo import summary
import os

from model import Model
from datahandler_auto import DataHandler
from train import train
from test import test
#----------------------------------------------------------------------

parser = argparse.ArgumentParser(description='RCGAN video prediction')

#Training/testing
parser.add_argument('--train', type=int, choices=[0, 1], default=1)
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--device', type=str, default='cpu') #of cuda voor GPU training

#Data
parser.add_argument('--train_path', type=str, default='data/auto-train-front-3-2.npy')
parser.add_argument('--valid_path', type=str, default='data/auto-validatie-front-3-2.npy')
parser.add_argument('--test_path', type=str, default='data/auto-test-front-3-2.npy')
parser.add_argument('--checkp_dir', type=str, default='checkpoints')
parser.add_argument('--results_dir', type=str, default='results')

#Images
parser.add_argument('--input_length', type=int, default=3)
parser.add_argument('--total_length', type=int, default=5)
parser.add_argument('--img_height', type=int, default=160)
parser.add_argument('--img_width', type=int, default=240)
parser.add_argument('--img_ch', type=int, default=1)

#Optimisation param
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--test_interval', type=int, default=20000)
parser.add_argument('--max_itr', type=int, default=80000)
parser.add_argument('--snapshot_interval', type=int, default=80000)
parser.add_argument('--num_save_samples', type=int, default=10)

args = parser.parse_args()
print(args)
print("")

#----------------------------------------------------------------------
# summary(model.generator, input_data=test)

if __name__ == '__main__':

    device = torch.device(args.device)

    print('Initializing networks\n')

    model = Model(args, device)

    if not os.path.isdir("results/"):
        os.mkdir("results/")

    if args.train == 1:

        traindata = DataHandler(args, args.train_path)
        validatiedata = DataHandler(args, args.valid_path)

        train(args, model, traindata, validatiedata, device)
    else:

        print("Starting test")

        testdata = DataHandler(args, args.test_path)

        model.generator = torch.load(args.pretrained_model)

        test(args, model, testdata, device)


