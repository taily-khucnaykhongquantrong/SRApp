import os

import torch
import numpy as np

# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from easydict import EasyDict as edict

from models.high2low.dataset import get_loader
from models.high2low.model import GEN_DEEP

# os.sys.path.append(os.getcwd())


def to_var(data):
    real_cpu = data
    batchsize = real_cpu.size(0)
    # Using real_cpu.cuda() if using GPU
    input = Variable(real_cpu)
    return input, batchsize


def test():
    torch.manual_seed(1)
    np.random.seed(0)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    opt = edict()
    opt.nGPU = 0
    opt.batchsize = 1
    # use CPU
    opt.cuda = False
    # cudnn.benchmark = True

    print("========================LOAD DATA============================")
    data_name = "img_align_celeba_png_set_1"
    test_loader = get_loader(data_name, opt.batchsize)
    net_G_low2high = GEN_DEEP(ngpu=opt.nGPU)
    # Remove this comment if using GPU with CUDA
    # net_G_low2high = net_G_low2high.cuda()
    a = torch.load("models/high2low/experiments/model.pkl", map_location="cpu")
    net_G_low2high.load_state_dict(a)
    net_G_low2high = net_G_low2high.eval()

    test_file = "img/sr/high2low/"

    if not os.path.exists(test_file):
        os.makedirs(test_file)

    print("Dataloader size: ", len(test_loader))

    for i, data_dict in enumerate(test_loader):
        print(i, data_dict["imgpath"], sep=" | ")

        data_low = data_dict["img16"]
        data_high = data_dict["img64"]
        img_name = os.path.basename(data_dict["imgpath"][0])

        data_input_low, batchsize_high = to_var(data_low)
        data_input_high, _ = to_var(data_high)
        data_high_output = net_G_low2high(data_input_low)

        path = os.path.join(test_file, img_name)
        vutils.save_image(data_high_output.data, path, normalize=True)


if __name__ == "__main__":
    test()
