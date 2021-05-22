import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse

#-------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Datasets maken voor predrnn en rcgan')

parser.add_argument('--network', type=str)
parser.add_argument('--seq_length', type=int, default=4)
parser.add_argument('--input_length', type=int, default=3)
parser.add_argument('--view', type=str, default='front')
parser.add_argument('--extension', type=str)
parser.add_argument('--add_dummy_data', type=int, default=0)

args = parser.parse_args()
print(args)

#-------------------------------------------------------------------------

SET = args.view #possible values: front - back - side - frontside - backside

BASEPATH = "/home/jef/Documents/data/"+SET+"view"

TRAINSET = 0.4 #percentage of dataset for training set 0.6
VALSET = 0.25 #persentage of dataset for validation set 0.15
TESTSET = 0.35 #percentage of dataset fot test set 0.25

LAATSTEVIER = False
EERSTEVIER = False

SEQGROOTTE = args.seq_length
PREDGROOTTE = SEQGROOTTE-args.input_length


def sorteren():
    """Functie sorteert alle afbeeldingen in de directory per model.
    Zodat we en sequentie per model hebben (in chronologische volgorde)"""

    used = []
    lijstsequenties = []

    for foto in os.listdir(BASEPATH):
        delen = foto.split('_')
        deel = str(delen[0]+'_'+delen[1]+'_')
        if deel in used:
            continue

        sequentie = []
        for f in os.listdir(BASEPATH):
            if deel in f:
                sequentie.append(f)

        sequentie = sorted(sequentie)

        lijstsequenties.append(sequentie)
        used.append(deel)
    
    return lijstsequenties

def verdeelsets(lijstsequenties):
    """Functie verdeelt de sequenties over een trainingsset,
    validatieset en testset"""

    new = []
    train = []
    val = []
    test = []

    for seq in lijstsequenties:
        if len(seq) >= SEQGROOTTE:
            new.append(seq) #enkel de sequneties die SEQGROOTTE of langer zijn bijhouden
    
    seqlen = len(new)

    numtrain = int(seqlen * TRAINSET)
    numval = int(seqlen * VALSET)
    numtest = int(seqlen * TESTSET)

    for seq in new:
        if len(seq) >= 15 and numtrain > 0:
            train.append(seq)
            numtrain -= 1
        elif numtest > 0:
            test.append(seq)
            numtest -= 1
        elif numval > 0:
            val.append(seq)
            numval -= 1
        elif numtrain > 0:
            train.append(seq)
            numtrain -= 1

    return train, val, test

def lengteseqaanpassen(lijstsequenties):
    """Functie zet grotere sequenties om naar kleinere sequenties 
    van een totale langte van SEQGROOTTE"""

    print("Voor:" + str(len(lijstsequenties)))

    newlist = []

    for seq in lijstsequenties:
        if len(seq) <= SEQGROOTTE:
            newlist.append(seq)
        else:
            if LAATSTEVIER == True:
                t = [seq[-4], seq[-3], seq[-2], seq[-1]]
                newlist.append(t)
            elif EERSTEVIER == True:
                t = [seq[0], seq[1], seq[2], seq[3]]
                newlist.append(t)
            else:
                for i in range((len(seq)-SEQGROOTTE+1)):
                    t = seq[i:i+SEQGROOTTE]
                    newlist.append(t)

    print("Na:" + str(len(newlist)))

    return newlist

def datasetvormgeven(lijstsequenties):
    """Functie maakt de benodigde lijsten om een zelfde vorm te krijgen
    als de moving mnist dataset die als voorbeeld gebruikt wordt.
    Afbeeldingen wordne in gelezen in grayscale en gecheckt dat de 
    breedte == 788 en de hoogte == 525. Als dit niet zo is worden deze
    afbeeldingen uit de dataset verwijdert."""

    inp = []
    gt = []
    imdata = []
    index = 0
    x = 0

    for element in lijstsequenties:

        tijdelijk = []
        errorinseq = False

        for model in element:
            img = cv2.imread(BASEPATH + '/' + model, cv2.IMREAD_GRAYSCALE)

            h, w = img.shape

            if 525 != h or 788 != w:
                # print("Probleem H of W bij " + model)
                errorinseq = True
                continue

            img = cv2.resize(img, (240, 164), interpolation = cv2.INTER_AREA)

            img = np.float32(img)

            img = cv2.normalize(img, img, 0, 1, norm_type=cv2.NORM_MINMAX)

            img = np.float16(img)

            tijdelijk.append([img])

        if errorinseq == True:
            # print("Sequentie geskipt!")
            continue

        if args.add_dummy_data != 0:
            for i in range(args.add_dummy_data):
                dummydata = np.zeros((164, 240), dtype=np.float16)
                tijdelijk.append([dummydata])
                
        for t in tijdelijk:
            imdata.append(t)

        totl = len(tijdelijk)
        inpl = SEQGROOTTE - PREDGROOTTE
        gtl = PREDGROOTTE + args.add_dummy_data

        tijdelijk = []
        tijdelijk.append(index)
        tijdelijk.append(inpl)

        inp.append(tijdelijk)

        index += inpl

        tijdelijk = []
        tijdelijk.append(index)
        tijdelijk.append(gtl)

        gt.append(tijdelijk)

        index += gtl

        x += 1

    print("Aantal sequenties = " + str(x))

    return inp, gt, imdata

def datasetvormgevenGAN(lijstsequenties):
    """Functie maakt de sequentielijst voor 
    gebruik te maken van het GAN netwerk"""

    data = []

    for element in lijstsequenties:
    
        tijdelijk = []
        errorinseq = False

        for model in element:
            img = cv2.imread(BASEPATH + '/' + model, cv2.IMREAD_GRAYSCALE)

            h, w = img.shape

            if 525 != h or 788 != w:
                # print("Probleem H of W bij " + model)
                errorinseq = True
                continue

            img = cv2.resize(img, (240, 160), interpolation = cv2.INTER_AREA)

            img = np.float32(img)

            img = cv2.normalize(img, img, 0, 1, norm_type=cv2.NORM_MINMAX)

            img = np.float16(img)

            tijdelijk.append([img])

        if errorinseq == True:
            # print("Sequentie geskipt!")
            continue
            
        data.append(tijdelijk)

    print("Aantal sequenties = %d" % (len(data)))
    
    return data

def opslaannpz(inp, gt, data, naam):
    """Functie slaagt de data op als een npz file, 
        in formaat:
        index --> lijst van inp en gt
        dims --> lijst met de dimensies
        data --> lijst van de afbeeldingen achter elkaar"""

    index = [inp, gt]
    dims = [[1, 164, 240]]
    np.savez("npz/auto-" + naam + "-" + SET + "-"+ args.extension +".npz", index=index, data=data, dims=dims)

def opslaanGAN(data, naam):
    """Functie zal de lijst van sequenties opslaan
    (dit zijn de sequenties die bedoelt zijn om te gebruiken voor de rcgan)"""

    np.save("gan/auto-" + naam + "-" + SET + "-"+ args.extension +".npy", data)


def main():

    print('Dataset maken van %s' % (BASEPATH))
    
    lijstsequenties = sorteren()

    train, val, test = verdeelsets(lijstsequenties)

    train = lengteseqaanpassen(train)
    val = lengteseqaanpassen(val)
    test = lengteseqaanpassen(test)


    if args.network == 'predrnn':
        print('predrnn')
        inp, gt, data = datasetvormgeven(train)
        opslaannpz(inp, gt, data, "train")

        inp, gt, data = datasetvormgeven(val)
        opslaannpz(inp, gt, data, "validatie")

        inp, gt, data = datasetvormgeven(test)
        opslaannpz(inp, gt, data, "test")

    elif args.network == 'rcgan':
        print('rcgan')
        data = datasetvormgevenGAN(train)
        opslaanGAN(data, 'train')

        data = datasetvormgevenGAN(val)
        opslaanGAN(data, 'validatie')

        data = datasetvormgevenGAN(test)
        opslaanGAN(data, 'test')

    else:
        print('Geen geldig netwerk geselecteerd')

if __name__ == '__main__':
    main()
