'''
1预测16有插值矫正
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from bipregan import *
import os
import numpy as np
from torch.utils.data import TensorDataset
from torchvision.utils import save_image
from dataloader1 import trainset
G = Generator().cuda()

porosity = np.load("pore1024.npy").reshape((-1,1))
print(porosity.shape)
porosity = torch.tensor(porosity).type(torch.FloatTensor).cuda()
data =trainset()
dataloader = torch.utils.data.DataLoader(data,batch_size=8)
mod = torch.load('HOU/G_250.pth')
G.load_state_dict(mod)
G.eval()
flag = 0
test = torch.zeros(1024*3,1024,1024)


with torch.no_grad():
    for img,porosity in dataloader:
        test[flag*8] = img[0].view(1024,1024)
        # print(img[flag*8:flag*8+7].shape)
        img = img.cuda()
        porosity  = porosity.cuda()
        ttt = img[0:4]
        res = G(img[0:4], porosity[1:8].view(7,1))
        res = torch.where(res<0.5,torch.zeros_like(res),torch.ones_like(res))
        test[flag*8+1:flag*8+8] = res.view(7,1024,1024)
        flag+=1
        print(flag)
        # for i in range(16):
        #     save_image(res[0, i].view(128, 128), 'res/test_%d.png' % (flag))
        #     save_image(img[0, i].view(128, 128), 'res/real_%d.png' % (flag))
        #     flag +=1
        #     print(flag)
test = test.data.cpu().numpy()
test.astype(np.uint8).tofile("res_niubi.raw")
