#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""
import pickle
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
        #transforms.Normalize(mean, std)
    ])
    #print("========+++++++++=====",shuffle)
    cifar100_testing = CIFAR100Test(path,transform=transform_test)
    #print("++++++++++++++++++++++++++",shuffle)
    #exit()i
    shuffle=True
    cifar100_test_loader = DataLoader(
        cifar100_testing, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return cifar100_test_loader
class CIFAR100Test():
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        #self.transform=transform
        path_test=path+"/*"
        
        test_files=glob.glob(path_test)
        x=[]
        y=[]
        name=[]
        #pickle输入修改0107
        for im in test_files:
            
          with open(im,"rb") as f:

            img = I.open(im).convert("L")

            img=np.array(img)
            img = img.transpose()
            # img=self.zuida_zuixiao_guiyihua(img)
            x.append(img)
            label=im.split("/")[-1].split("_")[0]
            y.append(int(label))
            name.append(im.split("/")[-1])

        x=np.array(x)
        self.label_test = np.array(y)
        #print(self.label_test)
        #exit()
        self.data_test = np.vstack(x).reshape(-1,1,settings.IMG_SIZE[0], settings.IMG_SIZE[1]) #1219尺寸修改
        print(self.data_test.shape,"data_test处理后的形状")
        self.image_test = self.data_test
        self.name_list = name

    def __len__(self):
        return self.data_test.shape[0]

    def __getitem__(self, index):
        label = self.label_test[index]
        image = self.image_test[index]
        name_list_ = self.name_list[index]
        image=torch.from_numpy(image)
        return image, label ,name_list_




def pingu(preds,labels,name,xinxi):
    preds=preds.numpy()
    labels=labels.numpy()
    
    for i,j,z in zip(preds,labels,name):
        if i != j:
            if str(i) +"_"+ str(j) not in xinxi:
                xinxi[str(i) +"_"+ str(j)] = 0
            xinxi[str(i) +"_"+ str(j)] += 1
            
            with open('data.txt', 'a') as f:
                f.write("name is %s"%z +"<=>" + "predict is %s"%i +"<=>" + "label is %s"%j +'\n')
    return xinxi
    

#def dabiaoqian():

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)
    #path = r'E:\basuo\data_new\fenlei_chonggou_12_30_yuan\12_30_yuan\test'  #地址测试集============
    #path = r'/data2/qcx/tianti_1_2_qcx/bashuo_pao/data/dataset0305/chongfenlei_jiangzao_28_87_MK1/test'
    path = settings.TEST_PATH
    #path = r'E:\python\wangting\pkl_jpg_zhankai_0110\pkl_jpg_zhankai_0109\test'
    #path = r"E:/python/wangting/new_tianti_data_12_18/pickle0107/test"
    cifar100_test_loader=get_test_dataloader(path,batch_size=args.b,num_workers=4)

    net.load_state_dict(torch.load(args.weights))
    #net=torch.load(args.weights, map_location=lambda storage, loc: storage)    #强制所有在单个gpu里面跑的程序在cpu里面跑

    net.eval()
    
    with torch.no_grad():
        net.eval()  
  
        test_loss = 0.0 # cost function error
        correct_tal = []
        xinxi={}
        for (images, labels , name) in cifar100_test_loader:
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()
            #print(labels,"====================================")
            #exit()
            outputs = net(images.float())
            _, preds = outputs.max(1)
            correct = preds.eq(labels).sum()
            correct_tal.append(correct)
            #print(correct,"查看有没有归一化===",len(cifar100_test_loader),"====================",images.shape)
            xinxi = pingu(preds,labels,name,xinxi)
        if args.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')
        print(correct_tal,"==================",np.sum(correct_tal))
        correct_tal=np.array(correct_tal)
        correct_sum = np.sum(correct_tal)
        
        print(xinxi)
        with open('data.txt', 'a') as f:
            f.write(args.weights + '\n')
            f.write("predict_label is "+"==>"+str(xinxi) + "\n")
            f.write("the acc is " + str(correct_sum / len(cifar100_test_loader.dataset)) + "\n")
    
        print('Evaluating Network.....')
        print('Test set:Accuracy: {:.4f}'.format(
            correct_sum / len(cifar100_test_loader.dataset),
        ))

