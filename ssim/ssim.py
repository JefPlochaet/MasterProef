import os
import cv2
import argparse
from skimage.metrics import structural_similarity 
from progress.bar import Bar

parser = argparse.ArgumentParser(description='RCGAN video prediction')

parser.add_argument('--path', type=str)
parser.add_argument('--input_length', type=int)
parser.add_argument('--total_length', type=int)

args = parser.parse_args()

def main():

    ssim = 0.0
    ssimprev = 0.0
    bar = Bar('Calculating SSIM', max=len(os.listdir(args.path)))

    for aantal, dir in enumerate(os.listdir(args.path)):

        newpath = args.path + "/" + dir
        gt = newpath + "/" + "gt" + str(args.input_length + 1) + ".png"
        pd = newpath + "/" + "pd" + str(args.input_length + 1) + ".png"
        gtprev = newpath + "/" + "gt" + str(args.input_length) + ".png"

        gtimg = cv2.imread(gt)
        pdimg = cv2.imread(pd)
        gtprevimg = cv2.imread(gtprev)

        ssim += structural_similarity(gtimg, pdimg, multichannel=True)
        ssimprev += structural_similarity(gtprevimg, pdimg, multichannel=True)

        bar.next()
    
    bar.finish()

    ssim = ssim/(aantal + 1)
    ssimprev = ssimprev/(aantal + 1)

    print("SSIM waarde voor gt en pd = " + str(ssim))
    print("SSIM waarde voor prevgt en pd = " + str(ssimprev))

if __name__ == '__main__':
    main()