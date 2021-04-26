import argparse
import torch
import functools
from torchinfo import summary

from model import Model
from datahandler_auto import DataHandler
from conversion import *
#----------------------------------------------------------------------

parser = argparse.ArgumentParser(description='RCGAN video prediction')

#Training/testing
parser.add_argument('--train', type=int, choices=[0, 1], default=1)
parser.add_argument('--pretrained_model', type=str, default='')

#Data
parser.add_argument('--train_path', type=str, default='data/auto-train-front-3-1.npy')
parser.add_argument('--valid_path', type=str, default='data/auto-validatie-front-3-1.npy')
parser.add_argument('--test_path', type=str, default='data/auto-test-front-3-1.npy')
parser.add_argument('--checkp_dir', type=str, default='checkpoints')
parser.add_argument('--results_dir', type=str, default='results')

#Images
parser.add_argument('--input_length', type=int, default=3)
parser.add_argument('--total_length', type=int, default=4)
parser.add_argument('--img_height', type=int, default=160)
parser.add_argument('--img_width', type=int, default=240)
parser.add_argument('--img_ch', type=int, default=1)

#Optimisation param
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--test_interval', type=int, default=20000)
parser.add_argument('--max_itr', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=1)
parser.add_argument('--snapshot_interval', type=int, default=80000)
parser.add_argument('--num_save_samples', type=int, default=10)

args = parser.parse_args()
print(args)

#----------------------------------------------------------------------
# summary(model.generator, input_data=test)

if __name__ == '__main__':

    device = torch.device('cuda')

    print('Initializing networks')

    model = Model(args)    

    generator = model.generator
    # generator.to(device)
    framediscriminator = model.discframe
    # framediscriminator.to(device)
    seqdiscriminator = model.discseq
    # seqdiscriminator.to(device)

    if args.train == 1:
        traindata = DataHandler(args, args.train_path)
        validatiedata = DataHandler(args, args.valid_path)
        traindata.begin()

        genoptim = torch.optim.Adam(params=generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        framediscoptim = torch.optim.Adam(params=framediscriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        seqdiscoptim = torch.optim.Adam(params=seqdiscriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    else:
        testdata = DataHandler(args, args.test_path)
    
    batch = traindata.get_batch()
    revbatch = traindata.get_revbatch()

    torchbatch = convert_inputseq(args, batch)
    # torchbatch.to(device)
    torchrevbatch = convert_inputseq(args, revbatch)
    # torchrevbatch.to(device)

    ngt = get_gt_frame(args, batch) #gt frame toekomst
    nsgt = get_gt_seq(args, batch)
    # ngt.to(device)
    mgt = get_gt_frame(args, revbatch) #gt frame verleden
    msgt = get_gt_seq(args, revbatch)
    # mgt.to(device)
    
    #train framediscriminator

    nacc = model.generator.forward(torchbatch) #genereer toekomstige frame
    macc = model.generator.forward(torchrevbatch) #genereer verleden frame (omgekeerde sequentie)

    torchbatchmacc = add_predframe_to_seq(args, torchbatch, macc) #voeg de genereerde frame toe aan de sequentie
    torchbatchnacc = add_predframe_to_seq(args, torchrevbatch, nacc) #voeg de genereerde frame toe aan de sequentie

    naccacc = model.generator.forward(torchbatchmacc) #genereer toekomstige frame gebaseerd op gegenereerde verleden frame
    maccacc = model.generator.forward(torchbatchnacc) #genereer verleden frame gebaseerd op de gegenereerde toekomstige frame

    # frameloss = model.frameloss(ngt, mgt, nacc, macc, naccacc, maccacc)

    # framediscoptim.zero_grad()
    # frameloss.backward()
    # framediscoptim.step()

    #train seqdiscriminator

    nacc = model.generator.forward(torchbatch) #genereer toekomstige frame
    macc = model.generator.forward(torchrevbatch) #genereer verleden frame (omgekeerde sequentie)

    torchbatchmacc = add_predframe_to_seq(args, torchbatch, macc) #voeg de genereerde frame toe aan de sequentie (vooraan)
    torchbatchnacc = add_predframe_to_seq(args, torchrevbatch, nacc) #voeg de genereerde frame toe aan de sequentie (vooraan)

    naccacc = model.generator.forward(torchbatchmacc) #genereer toekomstige frame gebaseerd op gegenereerde verleden frame
    maccacc = model.generator.forward(torchbatchnacc) #genereer verleden frame gebaseerd op de gegenereerde toekomstige frame

    nsacc = make_new_seq(args, nsgt, nacc)
    msacc = make_new_seq(args, msgt, macc)
    nsaccacc = make_new_seq(args, nsgt, naccacc)
    msaccacc = make_new_seq(args, msgt, maccacc)

    # seqloss = model.seqloss(nsgt, msgt, nsacc, msacc, nsaccacc, msaccacc)

    # seqdiscoptim.zero_grad()
    # seqloss.backward()
    # seqdiscoptim.step()

    #train generator

    nacc = model.generator.forward(torchbatch) #genereer toekomstige frame
    macc = model.generator.forward(torchrevbatch) #genereer verleden frame (omgekeerde sequentie)

    torchbatchmacc = add_predframe_to_seq(args, torchbatch, macc) #voeg de genereerde frame toe aan de sequentie (vooraan)
    torchbatchnacc = add_predframe_to_seq(args, torchrevbatch, nacc) #voeg de genereerde frame toe aan de sequentie (vooraan)

    naccacc = model.generator.forward(torchbatchmacc) #genereer toekomstige frame gebaseerd op gegenereerde verleden frame
    maccacc = model.generator.forward(torchbatchnacc) #genereer verleden frame gebaseerd op de gegenereerde toekomstige frame

    nsacc = make_new_seq(args, nsgt, nacc)
    msacc = make_new_seq(args, msgt, macc)
    nsaccacc = make_new_seq(args, nsgt, naccacc)
    msaccacc = make_new_seq(args, msgt, maccacc)

    imageloss = model.imageloss(ngt, mgt, nacc, macc, naccacc, maccacc)
    LoGloss = model.LoGloss(ngt, mgt, nacc, macc, naccacc, maccacc)
    frameloss = model.framelossGEN(nacc, macc, naccacc, maccacc)
    seqloss = model.framelossGEN(nacc, macc, naccacc, maccacc)
    totloss = imageloss + LoGloss + frameloss + seqloss

    genoptim.zero_grad()
    totloss.backward(torch.Tensor((args.batch_size, 1, args.img_height/8, args.img_width/8)))
    genoptim.step()

    #x = get_gt_seq(args, batch)

    #disctest = model.discframe.forward(ngt)

    # print(disctest.shape)