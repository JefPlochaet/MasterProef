import torch

from generator import GeneratorNetwork
from discriminator import DiscriminatorNetwork
from losses import *

class Model():
    def __init__(self, args):
        self.args = args

        self.generator = GeneratorNetwork(self.args)
        self.discframe = DiscriminatorNetwork(self.args, True)
        self.discseq = DiscriminatorNetwork(self.args, False)

        self.l1loss = torch.nn.L1Loss()
        self.LoG = LoG(kernel_size=5, sigma=0.65, in_channels=1)
        self.GANLossDisc = GANLossDisc()
        self.GANLossGen = GANLossGen()

    def imageloss(self, ngt, mgt, nacc, macc, naccacc, maccacc):
        """Image loss van de 6 paar afbeeldingen (zie paper)"""

        losseen = self.l1loss(mgt, macc)
        losstwee = self.l1loss(mgt, maccacc)
        lossdrie = self.l1loss(macc, maccacc)
        lossvier = self.l1loss(ngt, nacc)
        lossvijf = self.l1loss(ngt, naccacc)
        losszes = self.l1loss(nacc, naccacc)

        return losseen + losstwee + lossdrie + lossvier + lossvijf + losszes

    def LoGloss(self, ngt, mgt, nacc, macc, naccacc, maccacc): 
        """Laplacian of Gaussian loss van de 6 paar afbeeldingen (zie paper)"""

        losseen = self.l1loss(self.LoG(mgt), self.LoG(macc))
        losstwee = self.l1loss(self.LoG(mgt), self.LoG(maccacc))
        lossdrie = self.l1loss(self.LoG(macc), self.LoG(maccacc))
        lossvier = self.l1loss(self.LoG(ngt), self.LoG(nacc))
        lossvijf = self.l1loss(self.LoG(ngt), self.LoG(naccacc))
        losszes = self.l1loss(self.LoG(nacc), self.LoG(naccacc))

        return losseen + losstwee + lossdrie + lossvier + lossvijf + losszes

    def frameloss(self, ngt, mgt, nacc, macc, naccacc, maccacc):
        
        la1 = self.GANLossDisc(self.discframe(ngt), self.discframe(nacc))
        la2 = self.GANLossDisc(self.discframe(ngt), self.discframe(naccacc))
        la3 = self.GANLossDisc(self.discframe(mgt), self.discframe(macc))
        la4 = self.GANLossDisc(self.discframe(mgt), self.discframe(maccacc))

        return la1 + la2 + la3 + la4

    def seqloss(self, nsgt, msgt, nsacc, msacc, nsaccacc, msaccacc):

        lb1 = self.GANLossDisc(self.discseq(nsgt), self.discseq(nsacc))
        lb2 = self.GANLossDisc(self.discseq(nsgt), self.discseq(nsaccacc))
        lb3 = self.GANLossDisc(self.discseq(msgt), self.discseq(msacc))
        lb4 = self.GANLossDisc(self.discseq(msgt), self.discseq(msacc))

        return lb1 + lb2 + lb3 + lb4

    def framelossGEN(self, nacc, macc, naccacc, maccacc):

        la1 = self.GANLossGen(self.discframe(nacc))
        la2 = self.GANLossGen(self.discframe(naccacc))
        la3 = self.GANLossGen(self.discframe(macc))
        la4 = self.GANLossGen(self.discframe(maccacc))

        return la1 + la2 + la3 + la4

    def seqlossGEN(self, nsacc, msacc, nsaccacc, msaccacc):

        lb1 = self.GANLossGen(self.discseq(nsacc))
        lb2 = self.GANLossGen(self.discseq(nsaccacc))
        lb3 = self.GANLossGen(self.discseq(msacc))
        lb4 = self.GANLossGen(self.discseq(msaccacc))

        return lb1 + lb2 + lb3 + lb4
