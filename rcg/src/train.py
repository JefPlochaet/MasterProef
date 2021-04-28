import torch
import os
import cv2

from utils import *

def train(args, model, traindata, validatiedata):

    generator = model.generator
    generator.cuda()

    framediscriminator = model.discframe
    framediscriminator.cuda()

    seqdiscriminator = model.discseq
    seqdiscriminator.cuda()

    genoptim = torch.optim.Adam(params=generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    framediscoptim = torch.optim.Adam(params=framediscriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    seqdiscoptim = torch.optim.Adam(params=seqdiscriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    traindata.begin()

    gloss = []
    floss = []
    sloss = []

    for itr in range(1, 51):

        batch = traindata.get_batch()
        revbatch = traindata.get_revbatch()

        torchbatch = convert_to_tensor(args, batch)
        torchbatch.cuda()

        torchrevbatch = convert_to_tensor(args, revbatch)
        torchrevbatch.cuda()

        # -------------------
        # Get GT information
        # -------------------

        inseq = get_input_seq(args, torchbatch)
        revinseq = get_input_seq(args, torchrevbatch)

        ngt = get_gt_frame(args, torchbatch) #gt frame toekomst
        nsgt = get_gt_seq(args, torchbatch)
        
        mgt = get_gt_frame(args, torchrevbatch) #gt frame verleden
        msgt = get_gt_seq(args, torchrevbatch)
        
        # --------------------------
        # Train frame discriminator
        # --------------------------

        nacc = model.generator.forward(inseq) #genereer toekomstige frame
        macc = model.generator.forward(revinseq) #genereer verleden frame (omgekeerde sequentie)

        torchbatchmacc = add_front_frame(args, nsgt, macc) #voeg de genereerde frame toe aan de sequentie
        torchbatchnacc = add_front_frame(args, msgt, nacc) #voeg de genereerde frame toe aan de sequentie

        naccacc = model.generator.forward(get_input_seq(args, torchbatchmacc)) #genereer toekomstige frame gebaseerd op gegenereerde verleden frame
        maccacc = model.generator.forward(get_input_seq(args, torchbatchnacc)) #genereer verleden frame gebaseerd op de gegenereerde toekomstige frame

        frameloss = model.frameloss(ngt, mgt, nacc, macc, naccacc, maccacc)
        # frameloss = frameloss.squeeze()

        # print(frameloss.shape)
        # print(frameloss.mean().shape)

        framediscoptim.zero_grad()
        frameloss.mean().backward()
        framediscoptim.step()

        

        # -----------------------------
        # Train sequence discriminator
        # -----------------------------

        nacc = model.generator.forward(inseq) #genereer toekomstige frame
        macc = model.generator.forward(revinseq) #genereer verleden frame (omgekeerde sequentie)

        torchbatchmacc = add_front_frame(args, nsgt, macc) #voeg de genereerde frame toe aan de sequentie
        torchbatchnacc = add_front_frame(args, msgt, nacc) #voeg de genereerde frame toe aan de sequentie

        naccacc = model.generator.forward(get_input_seq(args, torchbatchmacc)) #genereer toekomstige frame gebaseerd op gegenereerde verleden frame
        maccacc = model.generator.forward(get_input_seq(args, torchbatchnacc)) #genereer verleden frame gebaseerd op de gegenereerde toekomstige frame

        nsacc = add_back_frame(args, nsgt, nacc)
        msacc = add_back_frame(args, msgt, macc)
        nsaccacc = add_back_frame(args, nsgt, naccacc)
        msaccacc = add_back_frame(args, msgt, maccacc)

        seqloss = model.seqloss(nsgt, msgt, nsacc, msacc, nsaccacc, msaccacc)

        seqdiscoptim.zero_grad()
        seqloss.mean().backward()
        seqdiscoptim.step()

        

        # ----------------
        # Train generator
        # ----------------

        nacc = model.generator.forward(inseq) #genereer toekomstige frame
        macc = model.generator.forward(revinseq) #genereer verleden frame (omgekeerde sequentie)

        torchbatchmacc = add_front_frame(args, nsgt, macc) #voeg de genereerde frame toe aan de sequentie
        torchbatchnacc = add_front_frame(args, msgt, nacc) #voeg de genereerde frame toe aan de sequentie

        naccacc = model.generator.forward(get_input_seq(args, torchbatchmacc)) #genereer toekomstige frame gebaseerd op gegenereerde verleden frame
        maccacc = model.generator.forward(get_input_seq(args, torchbatchnacc)) #genereer verleden frame gebaseerd op de gegenereerde toekomstige frame

        nsacc = add_back_frame(args, nsgt, nacc)
        msacc = add_back_frame(args, msgt, macc)
        nsaccacc = add_back_frame(args, nsgt, naccacc)
        msaccacc = add_back_frame(args, msgt, maccacc)

        imageloss = model.imageloss(ngt, mgt, nacc, macc, naccacc, maccacc)
        LoGloss = model.LoGloss(ngt, mgt, nacc, macc, naccacc, maccacc)
        frameloss = model.framelossGEN(nacc, macc, naccacc, maccacc)
        seqloss = model.framelossGEN(nacc, macc, naccacc, maccacc)
        totloss = imageloss + 0.005 * LoGloss + 0.003 * frameloss + 0.003 * seqloss

        genoptim.zero_grad()
        totloss.mean().backward()
        genoptim.step()

        print("itr%d" % (itr))

        # ----------
        # Validatie
        # ----------

        if(itr % 50 == 0):
            print("Validatietest itr%d" % (itr))

            validatiedata.begin(shuffle=False)
            
            batchid = 0

            if not os.path.isdir("results/" + str(itr)):
                os.mkdir("results/" + str(itr))

            while(validatiedata.batch_left() == True):

                if batchid < 1:
                    path = "results/" + str(itr) +  "/" + str(batchid)
                    if not os.path.isdir(path):
                        os.mkdir(path)

                    vbatch = validatiedata.get_batch()
                    vbatch.cuda()
                    
                    vbt = convert_to_tensor(args, vbatch)

                    genimg = generator.forward(get_input_seq(args, vbt))

                    img = genimg[0].detach().numpy()

                    img = np.uint8(img[0] * 256)
                    
                    print(img)

                    file = path + "/x.png"
                    cv2.imwrite(file, img)




                batchid += 1
                validatiedata.next()