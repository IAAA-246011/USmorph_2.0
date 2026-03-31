import os,glob
import shutil
from tqdm import tqdm

path = 'cae_img/'
targ_path = 'result_final/'
savepath = 'result_raw_50/'

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

cae_list = os.listdir(path)

for i in tqdm(glob.glob(targ_path+'*')):
    #print(i)
    #for j in i:
        #print(j)
    j_list = os.listdir(i)
    #print(j_list)
    for j_ in j_list:
        if j_ in cae_list:
            savepaths = mkdirs(savepath+i.split('/')[-1])
            shutil.copy(path+j_,savepaths+'/'+j_)
