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
    
        msetot = 0
        ssimtot = 0
        psnrtot = 0

        testdata.begin(shuffle=False)

        batchid = 0

        if not os.path.isdir(args.results_dir + "/test"):
            os.mkdir(args.results_dir + "/test")

        while(testdata.batch_left() == True):

            vbatch = testdata.get_batch()

            vbt = convert_to_tensor(args, vbatch).to(device)

            gtimg = get_gt_frame(args, vbt)
            pdimg = generator(get_input_seq(args, vbt))

            for i in range(args.batch_size):
                gtnp = np.uint8(gtimg[i, 0].detach().cpu() * 255)
                pdnp = np.uint8(pdimg[i, 0].detach().cpu() * 255)
                msetot += mean_squared_error(gtnp, pdnp)
                ssimtot += structural_similarity(gtnp, pdnp)
                psnrtot += peak_signal_noise_ratio(gtnp, pdnp)

            # save predication samples
            path = args.results_dir + "/test/" + str(batchid+1)
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
            testdata.next()

        mse = msetot/((batchid+1)*args.batch_size)
        ssim = ssimtot/((batchid+1)*args.batch_size)
        psnr = psnrtot/((batchid+1)*args.batch_size)

        print("ssim=%f\tmse=%f\tpsnr=%f\t" % (ssim, mse, psnr))
    
    generator.train()