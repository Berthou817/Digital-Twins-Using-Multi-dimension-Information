import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from dataloader import *
from bipregan import *
import os
import numpy as np
from torch.utils.data import TensorDataset
from torchvision.utils import save_image
from dataloader import *
lr=0.0002
beta_1=0.5
beta_2=0.999

G = Generator().cuda()

optim_G = optim.Adam(G.parameters(), lr=lr, betas=(beta_1, beta_2))

data = trainset()
mod = torch.load("HOU/G_250.pth")
G.load_state_dict(mod)
dataloader = torch.utils.data.DataLoader(data,batch_size=8)
epochs = 1000
mseloss = nn.MSELoss()
l1loss = nn.L1Loss()
GLoss_list = []

for ep in range(epochs):
    i = 251
    Gloss_list = []
    G.train()
    for img,porosity in dataloader:
        i = i+1
        img = img.cuda()
        porosity = porosity[1:8].view(7,1).cuda()

        fake_img = G(img[0:7], porosity)
        loss1 = mseloss(fake_img,img[1:8])
        loss2 = l1loss(fake_img,img[1:8])
        fake = torch.where(fake_img<0.5,torch.zeros_like(fake_img),torch.ones_like(fake_img))
        fake_pore =1- torch.mean(fake,dim=-1,keepdim=True).mean(dim=-2,keepdim=True).view(-1,1)
        loss3 = mseloss(fake_pore,porosity)
        loss3 = Variable(loss3,requires_grad=True)
        loss = loss1+loss2+loss3
        optim_G.zero_grad()
        loss.backward()
        optim_G.step()
        Gloss_list.append(loss.item())
        print(
            '[Epoch %d/%d] [Batch %d/%d] => 1_loss : %f / 2loss : %f / 3 loss : %f /loss : %f' \
            % (ep, epochs, i, len(dataloader), loss1.item(), loss2.item(), loss3.item(),loss.item()))

    GLoss_list.append(np.mean(np.array(Gloss_list)))

    if ep % 5 == 0:
            for num in range(7):
                save_image(fake_img[num].view(1024, 1024), 'HOU/pred_%d_%d.png' % (ep, num))
                save_image(img[num].view(1024, 1024), 'HOU/code_%d_%d.png' % (ep, num))
                # save_image(fake_img2[0],'photo1/test_single_%d.png' % (ep))
    if ep  % 10 == 0:
        np.savetxt("HOU/GLoss_%d.csv" % (ep), np.array(GLoss_list))

            #
        torch.save(G.state_dict(), "HOU/G_%d.pth" % (ep))


np.savetxt("HOU/GLoss.csv", np.array(GLoss_list))



torch.save(G.state_dict(), "HOU/G.pth")
