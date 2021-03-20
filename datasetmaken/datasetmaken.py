import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

SET = "frontside" #possible values: front - back - side - frontside - backside

BASEPATH = "/home/jef/Documents/data/"+SET+"view"

TRAINSET = 0.4 #percentage of dataset for training set 0.6
VALSET = 0.25 #persentage of dataset for validation set 0.15
TESTSET = 0.35 #percentage of dataset fot test set 0.25

LAATSTEVIER = True


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
        if len(seq) >= 4:
            new.append(seq) #enkel de sequneties die 4 of langer zijn bijhouden
    
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
    van een totale langte van 4 (3 input en 1 output)"""

    print("Voor:" + str(len(lijstsequenties)))

    newlist = []

    for seq in lijstsequenties:
        if len(seq) <= 4:
            newlist.append(seq)
        else:
            if LAATSTEVIER == False:
                for i in range((len(seq)-4+1)):
                    t = [seq[i], seq[i+1], seq[i+2], seq[i+3]]
                    newlist.append(t)
            else:
                t = [seq[-4], seq[-3], seq[-2], seq[-1]]
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
                print("Probleem H of W bij " + model)
                errorinseq = True
                continue

            img = cv2.resize(img, (240, 164), interpolation = cv2.INTER_AREA)

            img = np.float32(img)

            img = cv2.normalize(img, img, 0, 1, norm_type=cv2.NORM_MINMAX)

            img = np.float16(img)

            tijdelijk.append([img])

        if errorinseq == True:
            print("Sequentie geskipt!")
            continue
        
        for t in tijdelijk:
            imdata.append(t)

        totl = len(element)
        inpl = 3
        gtl = totl - inpl

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

def opslaannpz(inp, gt, data, naam):
    """Functie slaagt de data op als een npz file, 
        in formaat:
        index --> lijst van inp en gt
        dims --> lijst met de dimensies
        data --> lijst van de afbeeldingen achter elkaar"""

    index = [inp, gt]
    dims = [[1, 164, 240]]
    np.savez("npz/auto-" + naam + "-" + SET + "-l4.npz", index=index, data=data, dims=dims)


def main():
    
    lijstsequenties = sorteren()

    train, val, test = verdeelsets(lijstsequenties)

    # train = lengteseqaanpassen(train)
    # val = lengteseqaanpassen(val)
    test = lengteseqaanpassen(test)

    # inp, gt, data = datasetvormgeven(train)
    # opslaannpz(inp, gt, data, "train")

    # inp, gt, data = datasetvormgeven(val)
    # opslaannpz(inp, gt, data, "validatie")

    inp, gt, data = datasetvormgeven(test)
    opslaannpz(inp, gt, data, "test")

if __name__ == '__main__':
    main()
