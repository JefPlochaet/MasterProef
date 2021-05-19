import torch
import os
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

from utils import *

def test(args, model, testdata, device):

    generator = model.generator.to(device)

    generator.eval()

    with torch.no_grad():
    
        msetot = []
        ssimtot = []
        psnrtot = []

        for ind in range(args.total_length-args.input_length):
            msetot.append(0)
            ssimtot.append(0)
            psnrtot.append(0)

        testdata.begin(shuffle=False)

        batchid = 0

        if not os.path.isdir(args.results_dir + "/" + "test"):
            os.mkdir(args.results_dir + "/" + "test")

        while(testdata.batch_left() == True):

            vbatch = testdata.get_batch()

            vbt = convert_to_tensor(args, vbatch).to(device)

            pdimgs = []

            inp = get_input_seq(args, vbt, 0)

            for ind in range(args.total_length-args.input_length):
                gtimg = get_gt_frame(args, vbt, ind)
                pdimg = generator(get_input_seq(args, inp, ind))
                inp = torch.cat([inp, pdimg], dim=1)

                pdimgs.append(pdimg)

                for i in range(args.batch_size):
                    gtnp = np.uint8(gtimg[i, 0].detach().cpu() * 255)
                    pdnp = np.uint8(pdimg[i, 0].detach().cpu() * 255)
                    msetot[ind] += mean_squared_error(gtnp, pdnp)
                    ssimtot[ind] += structural_similarity(gtnp, pdnp)
                    psnrtot[ind] += peak_signal_noise_ratio(gtnp, pdnp)

            # save predication samples
            path = args.results_dir + "/" + "test" +  "/" + str(batchid+1)
            if not os.path.isdir(path):
                os.mkdir(path)

            for i in range(args.total_length):
                file = path + "/gt" + str(i+1) + ".png"
                img = np.uint8(vbt[0, i].detach().cpu() * 255)
                cv2.imwrite(file, img)

            for i in range(args.total_length-args.input_length):
                file = path + "/pd"+ str(args.input_length+1+i) +".png"
                img = np.uint8(pdimgs[i][0, 0].detach().cpu() * 255)    
                cv2.imwrite(file, img)

            batchid += 1
            testdata.next()

        for ind in range(args.total_length-args.input_length):
            mse = msetot[ind]/((batchid+1)*args.batch_size)
            ssim = ssimtot[ind]/((batchid+1)*args.batch_size)
            psnr = psnrtot[ind]/((batchid+1)*args.batch_size)

            print("frame%d: ssim=%f\tmse=%f\tpsnr=%f\t" % ((ind+1), ssim, mse, psnr))