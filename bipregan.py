import torch
import torch.nn as nn
from SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.model(x)
class DOWN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.model = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=4)

    def forward(self, x):
        return self.model(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=4)

    def forward(self, x):
        return self.up(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.total_length= 8
        self.input_length = 4
        cell_list1 = []
        for i in range(3):
            cell_list1.append(SpatioTemporalLSTMCell(16, 16, 256, 3, 1, 1))
        self.down1 = DOWN(2,16)
        self.cell_list1 = nn.ModuleList(cell_list1)
        cell_list2 = []
        for i in range(3):
            cell_list2.append(SpatioTemporalLSTMCell(64, 64, 64, 3, 1, 1))
        self.down2 = DOWN(16, 64)
        self.cell_list2 = nn.ModuleList(cell_list2)
        cell_list3 = []
        for i in range(3):
            cell_list3.append(SpatioTemporalLSTMCell(256, 256, 16, 3, 1, 1))
        self.down3 = DOWN(64, 256)
        self.cell_list3 = nn.ModuleList(cell_list3)
        cell_list4 = []
        for i in range(3):
            cell_list4.append(SpatioTemporalLSTMCell(128, 128, 64, 3, 1, 1))
        self.up3 = Up(256, 64)
        self.cell_list4 = nn.ModuleList(cell_list4)
        cell_list5 = []
        for i in range(3):
            cell_list5.append(SpatioTemporalLSTMCell(32, 32, 256, 3, 1, 1))
        self.up2 = Up(128, 16)
        self.cell_list5 = nn.ModuleList(cell_list5)
        self.up1 = Up(32, 1)
    def forward(self,x,z):
        next_frames = []
        target = x
        z = z.unsqueeze(dim=2).unsqueeze(dim=3)
        z = z.expand(z.size(0), z.size(1), x.size(2), x.size(3))
        h_t1=[]
        c_t1 = []
        h_t2 = []
        c_t2 = []
        h_t3 = []
        c_t3 = []
        h_t4 = []
        c_t4 = []
        h_t5 = []
        c_t5 = []
        for i in range(3):
            zeros = torch.zeros([1, 16, 256, 256]).cuda()
            h_t1.append(zeros)
            c_t1.append(zeros)
        for i in range(3):
            zeros = torch.zeros([1, 64, 64, 64]).cuda()
            h_t2.append(zeros)
            c_t2.append(zeros)
        for i in range(3):
            zeros = torch.zeros([1, 256, 16, 16]).cuda()
            h_t3.append(zeros)
            c_t3.append(zeros)
        for i in range(3):
            zeros = torch.zeros([1, 128, 64, 64]).cuda()
            h_t4.append(zeros)
            c_t4.append(zeros)
        for i in range(3):
            zeros = torch.zeros([1, 32, 256, 256]).cuda()
            h_t5.append(zeros)
            c_t5.append(zeros)
        memory = torch.zeros([1, 16, 256, 256]).cuda()
        for t in range(self.total_length - 1):
            if t < self.input_length:
                # print(target[t].shape, z[t].shape)
                inputs = torch.cat([target[t], z[t]],dim=0)
            else:
                inputs = torch.cat([out[0], z[t]])
            # 256
            x1 = self.down1(inputs.unsqueeze(dim=0))
            h_t1[0], c_t1[0], memory1 = self.cell_list1[0](x1, h_t1[0], c_t1[0], memory)
            for i in range(1, 3):
                h_t1[i], c_t1[i], memory1 = self.cell_list1[i](h_t1[i - 1], h_t1[i], c_t1[i], memory1)
            # 64
            h_t13, c_t13, memory2 = self.down2(h_t1[-1]),self.down2(c_t1[-1]),self.down2(memory1)
            h_t2[0], c_t2[0], memory2 = self.cell_list2[0](h_t13, h_t2[0], c_t2[0], memory2)
            for i in range(1, 3):
                h_t2[i], c_t2[i], memory2 = self.cell_list2[i](h_t2[i - 1], h_t2[i], c_t2[i], memory2)
            # 16
            h_t23, c_t23, memory3 = self.down3(h_t2[-1]), self.down3(c_t2[-1]), self.down3(memory2)
            h_t3[0], c_t3[0], memory3 = self.cell_list3[0](h_t23, h_t3[0], c_t3[0], memory3)
            for i in range(1, 3):
                h_t3[i], c_t3[i], memory3 = self.cell_list3[i](h_t3[i - 1], h_t3[i], c_t3[i], memory3)
            # 64
            h_t33, c_t33, memory4 = self.up3(h_t3[-1]), self.up3(c_t3[-1]), self.up3(memory3)
            h_t33, c_t33, memory4 = torch.cat([h_t33,h_t2[-1]],dim=1),torch.cat([c_t33,c_t2[-1]],dim=1),torch.cat([memory4,memory2],dim=1)
            h_t4[0], c_t4[0], memory4 = self.cell_list4[0](h_t33, h_t4[0], c_t4[0], memory4)
            for i in range(1, 3):
                h_t4[i], c_t4[i], memory4 = self.cell_list4[i](h_t4[i - 1], h_t4[i], c_t4[i], memory4)
            h_t43, c_t43, memory5 = self.up2(h_t4[-1]), self.up2(c_t4[-1]), self.up2(memory4)
            h_t43, c_t43, memory5 = torch.cat([h_t43, h_t1[-1]], dim=1), torch.cat([c_t43, c_t1[-1]], dim=1), torch.cat([memory5, memory1], dim=1)
            for i in range(1, 3):
                h_t5[i], c_t5[i], memory5 = self.cell_list5[i](h_t5[i - 1], h_t5[i], c_t5[i], memory5)
            out =self.up1(h_t5[-1])
            next_frames.append(out)
        next_frames = torch.stack(next_frames, dim=0).squeeze(dim=1)

        return next_frames


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

        self.short_cut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        )
        self.res_blocks = nn.Sequential(ResBlock(64, 128),
                                        ResBlock(128, 192),
                                        ResBlock(192, 256))
        self.pool_block = nn.Sequential(nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.AvgPool2d(kernel_size=8, stride=8, padding=0))

        # Return mu and logvar for reparameterization trick
        self.fc_mu = nn.Linear(1024, 1)
        self.fc_logvar = nn.Linear(1024, 1)

    def forward(self, x):
        # (N, 16, 128, 128) -> (N, 64, 64, 64)
        out = self.conv(x)
        # (N, 64, 64, 64) -> (N, 128, 32, 32) -> (N, 192, 16, 16) -> (N, 256, 8, 8)
        out = self.res_blocks(out)
        # (N, 256, 8, 8) -> (N, 256, 1, 1)
        out = self.pool_block(out)
        # (N, 256, 1, 1) -> (N, 256)
        out = out.view(x.size(0), -1)

        # (N, 256) -> (N, z_dim) x 2
        mu = self.fc_mu(out)
        log_var = self.fc_logvar(out)

        return (mu, log_var)

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'):
        super(ConvBlock, self).__init__()
        layers = []

        # Convolution Layer
        layers += [nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]

        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]

        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_block(x)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Discriminator with last patch (14x14)
        # (N, 3, 128, 128) -> (N, 1, 14, 14)
        self.d_1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False),
                                 ConvBlock(1, 32, k=4, s=4, p=1, norm=False, non_linear='leaky_relu'),
                                 # ConvBlock(32, 32, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 ConvBlock(32, 64, k=4, s=4, p=1, norm=True, non_linear='leaky-relu'),
                                 # ConvBlock(64, 64, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(64, 128, k=4, s=1, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(128, 1, k=4, s=1, p=1, norm=False, non_linear=None))

        # Discriminator with last patch (30x30)
        # (N, 16, 128, 128) -> (N, 1, 30, 30)
        self.d_2 = nn.Sequential(ConvBlock(1, 64, k=4, s=4, p=1, norm=False, non_linear='leaky_relu'),
                                 # ConvBlock(64, 64, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 ConvBlock(64, 128, k=4, s=4, p=1, norm=True, non_linear='leaky-relu'),
                                 # ConvBlock(128, 128, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(128, 256, k=4, s=2, p=0, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(256, 1, k=4, s=1, p=1, norm=False, non_linear=None)
                                 )

    def forward(self, x):
        out_1 = self.d_1(x)
        out_2 = self.d_2(x)
        return (out_1, out_2)
data  = torch.randn(7,1,1024,1024).cuda()
# z = torch.randn(15,1).cuda()
net = Encoder().cuda()
res,_ = net(data)
print(res.shape)