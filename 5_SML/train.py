
# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import argparse
import time
from datetime import datetime

import os,glob
import numpy as np
import shutil
from sklearn.model_selection import KFold
from tqdm import tqdm


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transcorms

from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def log(batch_index,stepnums,loss,train_acc_str,nums):
    train_log_filename = "log/%d/train_log1.txt"%nums
    train_log_filename1 = "log/%d/train_log2.txt"%nums
    train_log_filename2 = "log/%d/train_log3.txt"%nums
    train_log_txt_formatter = "{step:03d} {train_acc}\n"
    train_log_txt_formatter1 = "[step] {step:03d} [Loss is ] {loss_str} [train_acc is ] {train_acc}\n"
    to_write = train_log_txt_formatter.format(step = stepnums,
                                              train_acc=" ".join(["{}".format(train_acc_str)])
                                              )
    to_write1 = train_log_txt_formatter1.format(step = batch_index,
                                                loss_str=" ".join(["{}".format(loss)]),
                                                train_acc=" ".join(["{}".format(train_acc_str)])
                                               )
    if (stepnums%100)==0:
      to_write2 = train_log_txt_formatter1.format(step = stepnums,
                                                loss_str=" ".join(["{}".format(loss)]),
                                                train_acc=" ".join(["{}".format(train_acc_str)])
                                                  )
      with open(train_log_filename2, "a") as fii:
          fii.write(to_write2)
                                                 

    with open(train_log_filename, "a") as f:
        f.write(to_write)
    
    #with open(train_log_filename1, "a") as fi:
    #    fi.write(to_write1)




def log_test1(all_test_new1,nums):
    train_log_filename = "./log/%d/train_log1.txt"%nums
    train_log_txt_formatter = "[the test acc is] {step}\n"
    to_write = train_log_txt_formatter.format(step = all_test_new1)
    with open(train_log_filename, "a") as f:
        f.write(to_write)
    
def log_test(all_test_new,nums):
    train_log_filename1 = "./log/%d/train_log2.txt"%nums
    train_log_txt_formatter1 = "[the test acc is] {step}\n"
    to_write1 = train_log_txt_formatter1.format(step = all_test_new)
    with open(train_log_filename1, "a") as f:
        f.write(to_write1)

def log_test2(all_test_new2,nums):
    train_log_filename = "./log/%d/train_log4.txt"%nums
    train_log_txt_formatter = "[the test acc is] {step}\n"
    to_write = train_log_txt_formatter.format(step = all_test_new2)
    with open(train_log_filename, "a") as f:
        f.write(to_write)


def train_acc(outputs,labels,correct):
    #print(outputs.shape,"============",len(labels),"+++++++++++++++",correct)
    #print(outputs,"==-=-=-=-=-=")
    _, preds = outputs.max(1)
    correct = preds.eq(labels.long()).sum()
    
    train_1acc=correct/ len(labels)
    
    print("the train acc is ========%s"%train_1acc.cpu().numpy())
    return train_1acc.cpu().numpy()



@torch.no_grad()
def eval_100_steps(steps,nums,epoch=0, tb=True):


    start = time.time()
    net.eval()

    #test_loss = 0.0 # cost function error
    #correct = 0.0
    losslist=[]
    acclist = []
    
    for cifar100_test_loader in [cifar100_test_loader_0]:
    #for cifar100_test_loader in [cifar100_test_loader_0,cifar100_test_loader_1,cifar100_test_loader_2,cifar100_test_loader_3,cifar100_test_loader_4]:
        test_loss = 0.0
        correct = 0.0
        for (images, labels) in cifar100_test_loader:
            #test_loss = 0.0
            #correct = 0.0
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = net(images.float())

            #print(outputs.shape)

            loss = loss_function(outputs, labels.long())

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels.long()).sum()
            #exit()
        acclist.append(correct.float() / len(cifar100_test_loader.dataset))
        losslist.append(test_loss / len(cifar100_test_loader.dataset))
        finish = time.time()
    for i in [0]:   
        if args.gpu:
            print('GPU INFO.....')
            #print(torch.cuda.memory_summary(), end='')
            print('Evaluating Network.....')
            all_test_new1 = 'Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s , Testdata:{},The steps:{}'.format(
                        epoch,
                        losslist[i],
                        acclist[i],
                        finish - start,
                        checknum(i),
                        steps
                        )
            all_test_new = 'Epoch Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s, Testdata:{}'.format(
                        epoch,
                        losslist[i],
                        acclist[i],
                        finish - start,
                        checknum(i)
                        )

            #log_test1(all_test_new1)
            #log_test(all_test_new)
            log_test2(all_test_new1,nums)
            print(all_test_new)

            print('====%s===='%(checknum(i)))
        if tb:
            writer.add_scalar('Test/Average loss', losslist[i], epoch)
            writer.add_scalar('Test/Accuracy', acclist[i], epoch)

    return acclist[0],finish - start


def train(nums,epoch,times):

    start = time.time()
    #net.train()
    acc_list=[]
    sum_acc=[]
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        net.train()
        #print(images)
        #print(images.shape,'images.shape===')
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            
        #print(label==============================")
        #exit()
        optimizer.zero_grad()
        outputs = net(images.float())
        #print(outputs.shape,"output+++")
        loss = loss_function(outputs, labels.long())
        loss.backward()
        optimizer.step()
        correct=0
        train_acc_str=train_acc(outputs,labels,correct)
        acc_list.append(float(train_acc_str))
        
        #log(batch_index,stepnums,loss,train_acc_str)
        
        
        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
        stepnums = int(n_iter)
        log(batch_index,stepnums,loss,train_acc_str,nums)
        #print(n_iter,'==n_inter===',epoch,'===epoch===',len(cifar100_training_loader),'==len==',batch_index,'==batchidx====')

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)            
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}\tStepnums: {step_nums}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset),
            step_nums=stepnums
        ))            
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()
        accsum = sum(acc_list)
        sum_acc.append(accsum)
        acc_list=[]
        
   
        #0811-100steps-checking
        if (stepnums%settings.STEPS)==0:
            acc,test_time = eval_100_steps(stepnums,nums,epoch)
            acclog = str(acc).split(',')[0].split('(')[-1]
            finish_1 = time.time()
            costedtime = round((finish_1-start),4)
            print('The  steps: %s  The acc is :%s The costed time is:%s s'%((str(stepnums)),acclog,(str(costedtime))))
            all_test_new2 = ('The  steps: %s  The acc is :%s The costed time is:%s s'%((str(stepnums)),acclog,(str(costedtime))))    
            log_test2(all_test_new2,nums)
            
            if acc > 0.95:
                finish_acc = time.time()
                all_test_new2 = 'The target acc is:{:.4f}, Time consumed:{:.2f}s, Step is:{step}'.format(
                                acc,
                                finish_acc - times,
                                step = stepnums
                                )                
                log_test2(all_test_new2,nums)
            if (stepnums%10000)==0: 
                weights_step_path = checkpoint_step_path.format(net=args.net, step=stepnums, type='regular')
                print('saving weights file to {}'.format(weights_step_path))
                torch.save(net.state_dict(), weights_step_path)


    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)             

    
    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return accsum,sum_acc,finish - start


def checknum(i):
    if i == 0:        
        num = 'rotation_0'
    elif i == 1:
        num = 'rotation_80'
    elif i == 2:
        num = 'rotation_90'
    elif i == 3:
        num = 'rotation_180'
    elif i == 4:
        num = 'rotation_270'
    elif i == 5:
        num = 'rotation_100'
    return num

@torch.no_grad()
def eval_training(nums,epoch=0, tb=True):

    start = time.time()
    net.eval()

    #test_loss = 0.0 # cost function error
    #correct = 0.0
    losslist=[]
    acclist = []
    
    #选择测试集的角度值，默认是只选择不旋转的数据
    for cifar100_test_loader in [cifar100_test_loader_0]:
    #for cifar100_test_loader in [cifar100_test_loader_0,cifar100_test_loader_1,cifar100_test_loader_2,cifar100_test_loader_3,cifar100_test_loader_4]:
        test_loss = 0.0
        correct = 0.0
        for (images, labels) in cifar100_test_loader:
            #test_loss = 0.0
            #correct = 0.0
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = net(images.float())

            #print(outputs.shape)

            loss = loss_function(outputs, labels.long())

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels.long()).sum()
            #exit()
        acclist.append(correct.float() / len(cifar100_test_loader.dataset))
        losslist.append(test_loss / len(cifar100_test_loader.dataset))
        finish = time.time()
    for i in [0]:
    #for i in range(0,5):   
        if args.gpu:
            print('GPU INFO.....')
            #print(torch.cuda.memory_summary(), end='')
            print('Evaluating Network.....')
            all_test_new1 = 'Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s , Testdata:{}'.format(
                        epoch,
                        losslist[i],
                        acclist[i],
                        finish - start,
                        checknum(i)
                                                )
            all_test_new = 'Epoch Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s, Testdata:{}'.format(
                        epoch,
                        losslist[i],
                        acclist[i],
                        finish - start,
                        checknum(i)
                        )

            log_test(all_test_new,nums)
            #log_test1(all_test_new1)
            log_test2(all_test_new1,nums)
            print(all_test_new)

            print('====%s===='%(checknum(i)))

#add informations to tensorboard
        if tb:
            writer.add_scalar('Test/Average loss', losslist[i], epoch)
            writer.add_scalar('Test/Accuracy', acclist[i], epoch)

    return acclist[0],finish - start

    #return correct.float() / len(cifar100_test_loader.dataset),finish - start

class EarlyStopping:
   
    def __init__(self, patience, verbose=False, delta=0, path='./stop/weight-stop.pth', trace_func=print):
        if not os.path.exists('./stop/'):
          os.makedirs('./stop/')
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.acc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, acc, model):

        score = acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(acc, model)
            #self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            all_stop='The EarlyStopping counter: %s out of %s'%(str(self.counter),str(self.patience))
            log_test(all_stop,nums)
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter > self.patience:
                self.early_stop = True
        elif score >  self.best_score + self.delta:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.save_checkpoint(acc, model)
            self.counter = 0

    def save_checkpoint(self, acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.acc_max:.5f} --> {acc:.5f}).  Saving model ...')
            #self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        #self.val_loss_min = val_loss
        self.acc_max = acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)
    patience_num=settings.NUMS
    early_stopping = EarlyStopping(patience=patience_num, verbose=True) 


    #====10 zhe begin
    if not os.path.exists(settings.PATH + '/all/'):
        os.makedirs(settings.PATH + '/all/')

    # 设置数据路径
    data_dir = settings.PATH +  '/all'
    data_dir_ = settings.PATH


    # 获取所有图像的文件名
    img_names = os.listdir(data_dir)

    # 根据文件名排序，确保划分结果的一致性
    img_names.sort()

    # 创建一个表示图像编号的列表
    img_idx = list(range(len(img_names)))

    # 设置折数
    n_folds = 10

    # 初始化十折交叉验证器
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    nums = 0
    # 循环遍历所有的折
    for fold, (train_idx, val_idx) in enumerate(kf.split(img_idx)):
        nums+=1
        # 创建一个表示该折训练集图像文件名的列表
        train_names = [img_names[i] for i in train_idx]
        
        # 创建一个表示该折验证集图像文件名的列表
        val_names = [img_names[i] for i in val_idx]
        print(train_names[0],val_names[0])
        # 打印该折的训练集和验证集的大小
        print(f"Fold {fold+1}: Train size = {len(train_names)}, Val size = {len(val_names)}")
        # exit()
        # 在这里添加训练和验证的代码
        # ...

        if not os.path.exists('log/%s/'%nums):
            os.makedirs('log/%s/'%nums)


    #====10 zhe finish
        #data preprocessing:
        cifar100_training_loader = get_training_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=0,
            trainnames = train_names,
            batch_size=args.b,
            shuffle=True
        )
        

        cifar100_test_loader_0 = get_test_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            batch_size=args.b,
            num = 'test',
            testnames = val_names,
            num_workers=0,
            shuffle=True
            
        )
        
        cifar100_test_loader_1 = get_test_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            batch_size=args.b,
            #num = 'test',
            num = 'test_80',
            testnames = val_names,
            num_workers=0,
            shuffle=True
        )
        
        cifar100_test_loader_2 = get_test_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            batch_size=args.b,
            num = 'test_90',
            #num = 'test_png/test_90',
            testnames = val_names,
            num_workers=0,
            shuffle=True
        )
        
        cifar100_test_loader_3 = get_test_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            batch_size=args.b,
            num = 'test_180',
            #num = 'test_png/test_180',
            testnames = val_names,
            num_workers=0,
            shuffle=True
        )

        cifar100_test_loader_4 = get_test_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            batch_size=args.b,
            num = 'test_270',
            #num = 'test_png/test_270',
            testnames = val_names,
            num_workers=0,
            shuffle=True
        )
        

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
        iter_per_epoch = len(cifar100_training_loader)
        
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
        if args.resume:
            recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
            step_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_STEP_PATH, args.net), fmt=settings.DATE_FORMAT)
            if not recent_folder:
                raise Exception('no recent folder were found')
                
            if not step_folder:
                raise Exception('no recent folder were found')

            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
            checkpoint_step_path = os.path.join(settings.CHECKPOINT_STEP_PATH, args.net, recent_folder)

        else:
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
            checkpoint_step_path = os.path.join(settings.CHECKPOINT_STEP_PATH, args.net, settings.TIME_NOW)

        #use tensorboard
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        #since tensorboard can't overwrite old values
        #so the only way is to create a new tensorboard log
        print(os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))
        writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))
        input_tensor = torch.Tensor(16, 1, settings.IMG_SIZE[0], settings.IMG_SIZE[1])#0107按照pkl文件尺寸修改参数
        if args.gpu:
            input_tensor = input_tensor.cuda()
        writer.add_graph(net, input_tensor)

        #create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if not os.path.exists(checkpoint_step_path):
            os.makedirs(checkpoint_step_path)    
        
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
        checkpoint_step_path = os.path.join(checkpoint_step_path, '{net}-{step}-{type}.pth')
        
        best_acc = 0.0
        if args.resume:
            best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
            if best_weights:
                weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
                print('found best acc weights file:{}'.format(weights_path))
                print('load best training file to test acc...')
                net.load_state_dict(torch.load(weights_path))
                best_acc,times = eval_training(nums,tb=False)
                print('best acc is {:0.2f}'.format(best_acc))

            recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
            if not recent_weights_file:
                raise Exception('no recent weights file were found')
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
            print('loading weights file {} to resume training.....'.format(weights_path))
            net.load_state_dict(torch.load(weights_path))

            resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

        traintimes_sum=[]
        testtimes_sum=[]
        
        for epoch in range(1, settings.EPOCH + 1):
            start_time = time.time()
            if epoch > args.warm:
                train_scheduler.step(epoch)

            if args.resume:
                if epoch <= resume_epoch:
                    continue
            
            sum_acc,sum_acclist,sum_time = train(nums,epoch,times = start_time)
            traintimes_sum.append(sum_time)
            acc,test_time = eval_training(nums,epoch)
            testtimes_sum.append(test_time)
            # if args.earlystop:
            early_stopping(sum_acc,net)
            if early_stopping.early_stop:
                print('===Earlystopping===')
                stopping_time = sum(traintimes_sum)+sum(testtimes_sum)
                print('costing time is: %s'%str(stopping_time))
                break
                #exit()
            #start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_acc = acc
                continue

            if not epoch % settings.SAVE_EPOCH:
                
                
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)            


        writer.close()
