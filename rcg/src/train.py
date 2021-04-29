import torch
import os
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

from utils import *

def train(args, model, traindata, validatiedata):

    generator = model.generator
    # generator.cuda()

    framediscriminator = model.discframe
    # framediscriminator.cuda()

    seqdiscriminator = model.discseq
    # seqdiscriminator.cuda()

    genoptim = torch.optim.Adam(params=generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    framediscoptim = torch.optim.Adam(params=framediscriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    seqdiscoptim = torch.optim.Adam(params=seqdiscriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    traindata.begin(shuffle=True)

    gloss = []
    floss = []
    sloss = []

    for itr in range(1, args.max_itr+1):
        if traindata.batch_left() == False:
            traindata.begin(shuffle=True)

        batch = traindata.get_batch()
        revbatch = traindata.get_revbatch()

        torchbatch = convert_to_tensor(args, batch)
        # torchbatch.cuda()

        torchrevbatch = convert_to_tensor(args, revbatch)
        # torchrevbatch.cuda()

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

        framediscoptim.zero_grad()
        frameloss.mean().backward()
        framediscoptim.step()

        floss.append(frameloss.mean())

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

        sloss.append(seqloss.mean())

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

        gloss.append(totloss.mean())

        print("itr%d: gloss=%f\tfloss=%f\tsloss=%f" % (itr, gloss[itr-1], floss[itr-1], sloss[itr-1]))

        # ----------
        # Validatie
        # ----------

        if(itr % args.test_interval == 0):
            print("Validatietest itr:%d" % (itr))
            msetot = 0
            ssimtot = 0
            psnrtot = 0

            validatiedata.begin(shuffle=False)
            
            batchid = 0

            if not os.path.isdir(args.results_dir + "/" + str(itr)):
                os.mkdir(args.results_dir + "/" + str(itr))

            while(validatiedata.batch_left() == True):

                vbatch = validatiedata.get_batch()

                # vbt = convert_to_tensor(args, vbatch).cuda()
                vbt = convert_to_tensor(args, vbatch)

                gtimg = get_gt_frame(args, vbt)
                pdimg = generator.forward(get_input_seq(args, vbt))

                for i in range(args.batch_size):
                    gtnp = np.uint8(gtimg[i, 0].detach().cpu() * 255)
                    pdnp = np.uint8(pdimg[i, 0].detach().cpu() * 255)
                    msetot += mean_squared_error(gtnp, pdnp)
                    ssimtot += structural_similarity(gtnp, pdnp)
                    psnrtot += peak_signal_noise_ratio(gtnp, pdnp)

                # save predication samples
                if batchid < 10:
                    path = args.results_dir + "/" + str(itr) +  "/" + str(batchid+1)
                    if not os.path.isdir(path):
                        os.mkdir(path)

                    for i in range(args.total_length):
                        file = path + "/gt" + str(i+1) + ".png"
                        img = np.uint8(vbt[0, i].detach().cpu() * 255)
                        cv2.imwrite(file, img)

                    img = pdimg[0].detach().cpu().numpy()

                    img = np.uint8(img[0] * 255)

                    file = path + "/pd"+ str(args.input_length+1) +".png"
                    cv2.imwrite(file, img)

                batchid += 1
                validatiedata.next()
            
            mse = msetot/((batchid+1)*args.batch_size)
            ssim = ssimtot/((batchid+1)*args.batch_size)
            psnr = psnrtot/((batchid+1)*args.batch_size)

            print("ssim=%f\tmse=%f\tpsnr=%f\t" % (ssim, mse, psnr))
        
        # save model
        if(itr % args.snapshot_interval == 0):
            if not os.path.isdir(args.checkp_dir):
                os.mkdir(args.checkp_dir)

            torch.save(generator.state_dict(),  args.checkp_dir + "/generator_param_"+ str(itr) +".pkl")
            torch.save(framediscriminator.state_dict(),  args.checkp_dir + "/framediscriminator_param_"+ str(itr) +".pkl")
            torch.save(seqdiscriminator.state_dict(),  args.checkp_dir + "/seqdiscriminator_param_"+ str(itr) +".pkl")