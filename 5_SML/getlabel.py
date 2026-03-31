# -*- coding: UTF-8 -*-
import pickle,shutil,os
import argparse
import glob
from matplotlib import pyplot as plt
from PIL import Image as I
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from conf import settings
from utils import get_network#, get_test_dataloader

def get_test_dataloader(path,batch_size, num_workers=0, shuffle=False):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    cifar100_testing = CIFAR100Test(path,transform=transform_test)
    shuffle=True
    cifar100_test_loader = DataLoader(
        cifar100_testing, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar100_test_loader

class CIFAR100Test():
    def __init__(self, path, transform=None):
        path_test=path+"/*"
        test_files=glob.glob(path_test)
        x=[]
        name=[]
        for im in test_files:
            img = I.open(im).convert("L")
            img=np.array(img)
            img = img.transpose()
            x.append(img)
            name.append(im.split("/")[-1])
        x=np.array(x)
        self.data_test = np.vstack(x).reshape(-1, 1, settings.IMG_SIZE[0], settings.IMG_SIZE[1])
        print(self.data_test.shape,"data_test处理后的形状")
        self.image_test = self.data_test
        self.name_list = name

    def __len__(self):
        return self.data_test.shape[0]

    def __getitem__(self, index):
        image = self.image_test[index]
        name_list_ = self.name_list[index]
        image=torch.from_numpy(image)
        return image, name_list_

if __name__ == '__main__':    
    k = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()
    savename = str(args)
    savename = savename.split('/')[-1].split('-')[0]
    net = get_network(args)
    path = settings.NO_LABEL_PATH
    #打完标签后的文件输出储存地址,根据使用的模型文件
    save_path = r"%s_predict_raw/"%savename
    cifar100_test_loader=get_test_dataloader(path,batch_size=args.b,num_workers=4)
    net.load_state_dict(torch.load(args.weights))
    

    net.eval()
    for aa in range(5):
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path) 
            
        if os.path.isdir(save_path + str(aa)) == False:
            os.mkdir(save_path + str(aa)) 
    
    with torch.no_grad():
        net.eval()  
        for (images,name) in cifar100_test_loader:
            if args.gpu:
                images = images.cuda()
            outputs = net(images.float())
            _, preds = outputs.max(1)
            pred = preds.cpu().tolist()
            for la,na in zip(pred,name):
                na = na.split("/")[-1]
                
                k = k + 1
                print(save_path + str(la) +"/" + na,k,'========')
                #将判断后的无标签图像储存到相应标签对应的文件夹
                img = I.open(path_1 + na).convert('L')
                img = img.save(save_path + str(la) +"/" + na)
            
            
            
