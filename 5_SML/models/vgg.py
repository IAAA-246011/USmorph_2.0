"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
import torch
import torch.nn as nn


cfg = {
    'A' : [80],
    'B' : [72],
    'C' : [64],
    'D' : [56], 
    'E' : [80],
    'F' : [72],
    'G' : [64],
    'H' : [56]

       }
#1219卷积核尺寸修改，100-28，90-80
cfg_conv = {
          "A_conv" : [28,3],
          "B_conv" : [28,5],
          "C_conv" : [28,7],
          "D_conv" : [28,9],
          "E_conv" : [28,30],
          "F_conv" : [28,50],
          "G_conv" : [28,70],
          "H_conv" : [28,80]

            }

class VGG(nn.Module):

    def __init__(self, features,features_1, features_2, features_3,features_4,features_5, features_6, features_7,num_class=5):
        super().__init__()
        self.features = features
        self.features_1 = features_1
        self.features_2 = features_2
        self.features_3 = features_3
        self.features_4 = features_4  #4-7为1211新改动
        self.features_5 = features_5
        self.features_6 = features_6
        self.features_7 = features_7
        #print(features_1,'features_1')
        #exit()
        self.juanji = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(80,5), padding=0))    #报错处：（1，1，180，5）
        self.juanji1 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(72,5), padding=0))
        self.juanji2 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(64,5), padding=0))
        self.juanji3 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(56,5), padding=0))
        self.juanji4 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(80,5), padding=0))
        self.juanji5 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(72,5), padding=0))
        self.juanji6 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(64,5), padding=0))
        self.juanji7 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(56,5), padding=0))
        self.classifier = nn.Sequential(
            nn.Linear(418, 4096),      #1219 418是卷积核最后输出的值，改成这个就好
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),#加了两层
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        #print(x.shape) torch.Size([16, 1, 100, 314])
        #exit()
        
        output = self.features(x)
        output1 = self.features_1(x)
        output2 = self.features_2(x)
        output3 = self.features_3(x)
        output4 = self.features_4(x)#4-7为1211新改动
        output5 = self.features_5(x)
        output6 = self.features_6(x)
        output7 = self.features_7(x)
        #print('self.feature=================')
        #print(output.shape)  torch.Size([16, 80, 1, 312])疑似报错处
        #exit()
        output=torch.squeeze(output,dim=2)
        output1=torch.squeeze(output1,dim=2)
        output2=torch.squeeze(output2,dim=2)
        output3=torch.squeeze(output3,dim=2)
        output4=torch.squeeze(output4,dim=2)#4-7为1211新改动
        output5=torch.squeeze(output5,dim=2)
        output6=torch.squeeze(output6,dim=2)
        output7=torch.squeeze(output7,dim=2)
        #print('torch.squeeze===============')
        #print(output.shape)  torch.Size([16, 80, 312])
        output=torch.unsqueeze(output, dim=1)
        output1=torch.unsqueeze(output1, dim=1)
        output2=torch.unsqueeze(output2, dim=1)
        output3=torch.unsqueeze(output3, dim=1)
        output4=torch.unsqueeze(output4, dim=1)#4-7为1211新改动
        output5=torch.unsqueeze(output5, dim=1)
        output6=torch.unsqueeze(output6, dim=1)
        output7=torch.unsqueeze(output7, dim=1)
        #print('torch.unsqueeze===============')
        #print(output.shape)   torch.Size([16, 1, 80, 312])疑似报错

        #print(output.shape,"=+++++++++++++====================================",output1.shape,"-=-=",output2.shape,"------",output3.shape)
        output = self.juanji(output)
        output1 = self.juanji1(output1)
        output2 = self.juanji2(output2)
        output3 = self.juanji3(output3)
        output4 = self.juanji4(output4)#4-7为1211新改动
        output5 = self.juanji5(output5)
        output6 = self.juanji6(output6)
        output7 = self.juanji7(output7)
        
        #print('self.juanji===============')
        #print(output.shape)
        
        output = output.view(output.size()[0], -1)
        output1 = output1.view(output1.size()[0], -1)
        output2 = output2.view(output1.size()[0], -1)
        output3 = output3.view(output1.size()[0], -1)
        output4 = output4.view(output1.size()[0], -1)#4-7为1211新改动
        output5 = output5.view(output1.size()[0], -1)
        output6 = output6.view(output1.size()[0], -1)
        output7 = output7.view(output1.size()[0], -1)
        #print('output.view===============')
        #print(output.shape)    torch.Size([16, 308])


        tal_output=torch.cat((output,output1,output2,output3,output4,output5,output6,output7), dim=1)
        #print('tal_output.shape===============',tal_output.shape)
        #exit()
        real_output=self.classifier(tal_output)
        #print(real_output.shape)   torch.Size([16, 5])
        #exit()
        return real_output

def make_layers(cfg,cfg_conv, batch_norm=True):
    layers = []

    input_channel = 1
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            
            
            
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=(cfg_conv[0],cfg_conv[1]), padding=0)]
        
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'],cfg_conv["A_conv"],batch_norm=True),make_layers(cfg['B'],cfg_conv["B_conv"],batch_norm=True),make_layers(cfg['C'],cfg_conv["C_conv"],batch_norm=True),make_layers(cfg['D'],cfg_conv["D_conv"],batch_norm=True),make_layers(cfg['E'],cfg_conv["E_conv"],batch_norm=True),make_layers(cfg['F'],cfg_conv["F_conv"],batch_norm=True),make_layers(cfg['G'],cfg_conv["G_conv"],batch_norm=True),make_layers(cfg['H'],cfg_conv["H_conv"],batch_norm=True))
#E-G为1211新改动
#备忘
#,make_layers(cfg['E'],cfg_conv["E_conv"],batch_norm=True),make_layers(cfg['F'],cfg_conv["F_conv"],batch_norm=True)
def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))
