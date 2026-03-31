#!-*-coding:utf-8-*-
# 获得图片的label2path

import glob
import pickle
import time 
import random
from PIL import Image
import numpy as np
import config as C


class data_utils():
    def __init__(self):
        self.x=C.pix_x
        self.y=C.pix_y
        self.get_label2path()
        self.jth_batch=0

    def get_label2path(self):    
        st=time.time()
        pool = []
        #dizhi = "/data2/duqimeng/8_Apackage_classif/1_dataset/image_daiyao_28/cut_image"
        total_file = 0
        for file_ in glob.glob(C.path+"/*"):
            pool.append(file_)
            total_file +=1
        print("共计%s个样本，其中"%(total_file))
        self.pool=pool
        #random.shuffle(self.pool)
        self.total_file=total_file

    def give_batch(self,batch_size,shuffle=False):
        random.shuffle(self.pool)
        path_lst=self.pool[0:batch_size]
        ret=self.give_feature(path_lst)
        return ret
        
    def give_feature(self,path_lst):
        features=[]
        for path_ in path_lst:
           # print(path_)
           # exit()
            img=Image.open(path_).convert("L").resize((C.pix_x,C.pix_y))
            img = np.array(img)
            features.append(img)

        return features
        
    def norm(self,data,max_,min_):
        ma = np.max(data)
        mi = np.min(data)
        ret = (data-mi)/(ma-mi)
        ret=ret*(max_)
        ret=ret.astype(int)
        return ret

    def norm_(self,data,max_,min_):
        mean = np.mean(data)
        std = np.std(data)
        ret = (data-mean)/std
        ret=ret*(max_/3)
        ret=ret.astype(int)
        ret[ret < min_] = min_
        ret=ret -min_

        return ret


if __name__=="__main__":
    Data=data_utils()
    path,batch=Data.give_batch_reconstruct(100)
    print(path)
    print(len(batch[1]),len(batch[1][0]))
    print(np.array(batch).shape)#N*x*y
    #print(np.array(batch).shape)
















