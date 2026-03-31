""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

PATH = '/home/songjie/share/machine_learning/sml_method/data'
TEST_PATH = '/home/songjie/share/machine_learning/sml_method/data'
NO_LABEL_PATH = '/home/songjie/share/machine_learning/sml_method/nolabel_data'

#极坐标尺寸设置，可根据实际尺寸修改
IMG_SIZE = [56,128]
#原始图像尺寸设置，可根据实际尺寸修改
# IMG_SIZE = [28,28]
#早停参数，若准确率在150轮后无增长训练将自动停止
NUMS = 150

#测试的间隔步数
STEPS = 100


#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'
CHECKPOINT_STEP_PATH = 'checkpoint_step'
#total training epoches
EPOCH = 2000
MILESTONES = [60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








