import numpy as np
import the_net as m
import utils 
import random
import logging
import config as CF
import tensorflow as tf
import time
import argparse
import glob
import os
import cv2
from keras.backend.tensorflow_backend import set_session
import keras
from PIL import Image
parser = argparse.ArgumentParser()
parser.description='训练还是生成bp'
parser.add_argument("--train", default=False,help="是否训练",type=bool)
parser.add_argument("--savePB", default=False,help="储存ckpt",type=bool)
parser.add_argument("--test", default=False,help="生成降噪图像",type=bool)
parser.add_argument("--encode", default=False,help="进行编码",type=bool)
args = parser.parse_args()

is_train=args.train
is_savePB=args.savePB
is_test=args.test
is_encode=args.encode
if not is_train and not is_savePB and not is_test and not is_encode :
   print("请使用 python main.py --train True,或者 python main.py --savePB True")
   exit()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s :%(message)s",
                    datefmt="%Y-%m_%d %H:%M:%S",
                    filename=CF.logging_name,
                    filemode="w")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


#config = tf.ConfigProto()  
#config.gpu_options.allow_growth = True 
#sess = tf.Session(config=config)
#set_session(sess)
#keras.backend.clear_session()

tf.reset_default_graph()
sess=tf.Session()
model=m.CAE(sess=sess,logging=logging)
#model_1.print_var()
Data=utils.data_utils()

if is_train:
    st=time.time()
    for k in range(CF.maxstep):
        batch=Data.give_batch(CF.batch_size)
        batch = np.array(batch)
        is_show=True if (k+1)%CF.step_show==0 else False
        is_save=True if (k+1)%CF.step_save==0 else False
        if is_show:
            print("用时%s秒"%(time.time()-st))
            st=time.time()
        model.train(batch, is_show,is_save)
 
if is_test:
    a_dizhi = glob.glob("fit_img/*")
    for ii in range(int(len(a_dizhi)/3)):
       dizhi_suoyin = a_dizhi[ii*3:(ii+1)*3]
       all_img = []
       for jj in dizhi_suoyin: 
           img=Image.open(jj).convert("L")
           img = np.array(img)
           all_img.append(img) 
       batch=all_img
       jiangzaojieshu=model.jiangzao_shuchu(batch)
       
       for ooo,uuu in zip(dizhi_suoyin,jiangzaojieshu):
           zhen_ooo = "cae_img/" + ooo.split("/")[-1]
           cv2.imwrite(zhen_ooo,uuu)
               
if is_savePB:
    model.savePB()
