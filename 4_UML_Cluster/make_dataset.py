import numpy as np
import PIL.Image as I
from tqdm import tqdm
# import config as C
import glob,shutil,os

def make_lab(label):
    if label == 'UNC':
        labels = 0
    if label == 'LTD':
        labels = 1
    if label == 'IRR':
        labels = 2
    if label == 'ETD':
        labels = 3
    if label == 'SPH':
        labels = 4
    return str(labels)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    print('start making dataset...')
    name = {}
    with open('result.txt','r',encoding='utf-8') as f:
        for i in f:
            i=i.strip().split()
            # print(i[0],type(i[0]))
            # exit()
            vals = make_lab(i[1])
            name.update({i[0]:vals})
    for j in tqdm(name):
        path = 'result_img/%s/'%j
        filelist = os.listdir(path)
        savepath = 'dataset/'
        savepath = make_dir(savepath)
        for i in filelist:
            shutil.copy(path+i,savepath+name[j]+'_'+i)
        # print(type(j))
        # exit()
    # print(name)