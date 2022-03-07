from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
import torch

data_path = os.listdir("train")
data_path.sort(key=lambda x:int(x[:-4]))

def default_loader(path):
    data_pil =  np.load("train/%s"%(path)).reshape((1,1024,1024))
    data_pil = np.where(data_pil == 1, np.ones_like(data_pil), np.zeros_like(data_pil))
    data_tensor = torch.tensor(data_pil).type(torch.FloatTensor)
    return data_tensor
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        self.images = data_path
        self.loader = loader
    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        # target = 1-torch.mean(img,keepdim=True,dim=-1).mean(keepdim=True,dim=-2).view(-1,1)
        target = 1-torch.mean(img)
        return img,target
    def __len__(self):
        return len(self.images)
# test=torch.zeros(8,1024,1024)
# train_data  = trainset()
# trainloader = DataLoader(train_data, batch_size=1)
# flag = 0
# for i,j in trainloader:
#     if flag<8:
#         test[flag] = i.view(1024,1024)
#         flag += 1
#         print(flag)
# test = test.data.cpu().numpy()
# test.astype(np.uint8).tofile("test8.raw")
# np.save("pore1024.npy",test)



