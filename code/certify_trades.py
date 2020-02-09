# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
import numpy as np
from models.wideresnet import *
from archs.cifar_resnet import resnet as resnet_cifar
from tqdm import tqdm
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--imagesize', type=int, default=32, help='input image size')
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=1000)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument("--load-dir", type=str, help="loading directory")
args = parser.parse_args()


# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}



def mask_maker(img):
    """
    The function outputs an all-1 mask. One can adapt the function to creating other masks.
    """
    h = img.size(1)
    w = img.size(2)


    mask = np.ones((h, w), np.float32)

    mask = torch.from_numpy(mask)
    mask = mask.repeat(3, 1, 1)

    return mask


if __name__ == "__main__":
    # load the base classifier
    # base_classifier = WideResNet().cuda()
    base_classifier = resnet_cifar(depth=110, num_classes=10).to(device)
    base_classifier = nn.DataParallel(base_classifier)
    base_classifier.load_state_dict(torch.load(args.load_dir))

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split, args.imagesize)
    for i in tqdm(range(len(dataset))):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        # certify the prediction of g around x
        before_time = time()
        x = x.to(device)
        mask = mask_maker(x).to(device)
        prediction, radius = smoothed_classifier.certify(x, mask, args.N0, args.N, args.alpha, args.batch)
        radius = radius/np.sqrt(mask.cpu().sum())
        after_time = time()
        correct = int(prediction == label)
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
