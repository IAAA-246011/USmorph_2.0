import config as CF
import glob
import os,shutil
import numpy as np
import pandas as pd


def count(lab,same_0,same_1,same_2,same_3,same_4,same_5,same_6,same_7):
    if lab == 0:
        same_0+=1
    if lab == 1:
        same_1+=1
    if lab == 2:
        same_2+=1
    if lab == 3:
        same_3+=1
    if lab == 4:
        same_4+=1
    if lab == 5:
        same_5+=1
    if lab == 6:
        same_6+=1
    if lab == 7:
        same_7+=1
    return same_0,same_1,same_2,same_3,same_4,same_5,same_6,same_7  
def check_max(d):
    d1 = {}
    alls = []
    if len(d) !=0:
        for k,v in d.items():
            d1.update({str(v):k})
            alls.append(v)
        try:
            alls = np.array(alls)
            # print(alls)
            # exit()
            lab = np.max(alls)
        except:
            # label=0
            print(alls,'====')
            try:
                lab = alls[0]
                
                # print(label,'label')
                # label = int(d1[str(lab)])
            except:
                print('文件是空的，有内鬼')
                # lab = 0
                pass
                # label = 0#int(d1[str(0)])
                # pass
                # label=0
        label = int(d1[str(lab)])
        # labels = label
    else:
        label = 8
        pass
    return label

def check_file(path):
    if os.path.exists(path):
        pass
        # shutil.rmtree(path)
    else:
        os.makedirs(path)
    return path

def count_error(k,lab,num):
    name = 'log/data_%s/'%num
    labs = (k.split('/')[-1].split('_')[0])
    savepath = check_file(name+'%s/'%str(lab))
    shutil.copy(k,savepath+k.split('/')[-1])

def count_right(k,lab,num):
    name = '../SML/dataset_%s/'%num
    labs = (k.split('/')[-1].split('_')[0])
    savepath = check_file(name+'%s/'%str(lab))
    shutil.copy(k,savepath+k.split('/')[-1])


class_num = int(input('请输入数据类别总数：'))
m_name = input('please certain methord(pca/raw)：')
if m_name== 'pca':
    d_name = 'pca'
else:
    d_name = 'raw'
path = 'result_final%s/'%class_num
# for i in glob.glob(path+'/*'):
all_correct = 0
all_num = 0
nums = 0
nums1 = 0
if not os.path.exists('log/%s/'%d_name):
    os.makedirs('log/%s/'%d_name)
if os.path.exists('log/%s/check_result_%s'%(d_name,class_num)):
    os.remove('log/%s/check_result_%s'%(d_name,class_num))
if os.path.exists('log/%s/all_result_%s'%(d_name,class_num)):
    os.remove('log/%s/all_result_%s'%(d_name,class_num))
if os.path.exists('log/%s/right_result_%s'%(d_name,class_num)):
    os.remove('log/%s/right_result_%s'%(d_name,class_num))
if os.path.exists('log/%s/result_%s'%(d_name,class_num)):
    os.remove('log/%s/result_%s'%(d_name,class_num))
same_0 = 0
same_1 = 0
same_2 = 0
same_3 = 0
same_4 = 0
same_5 = 0
same_6 = 0
same_7 = 0

diff_0 = 0
diff_1 = 0
diff_2 = 0
diff_3 = 0
diff_4 = 0
diff_5 = 0
diff_6 = 0
diff_7 = 0


all_0 = 0
all_1 = 0
all_2 = 0
all_3 = 0
all_4 = 0
all_5 = 0
all_6 = 0
all_7 = 0

for j in glob.glob(path+'/*'):
    nums+=len(glob.glob(j+'/*'))
    # print(j)
    # exit()
    if j.split('/')[-1] == 'rest':
        nums1 +=len(glob.glob(j+'/*'))
        pass
    else:
    
        alls = []
        alls_name = []
        for k in glob.glob(j+'/*'):
            alls.append(int(k.split('/')[-1].split('_')[0]))
            alls_name.append(k)
        labels = list(set(alls))
        label = {}
        num = {}#{'real':0,'predict':0,'total':0}
        for i_ in labels:
            label.update({str(i_):0})
            num.update({str(i_):0})
        for k in glob.glob(j+'/*'):
            print(k.split('/')[-1].split('_')[0],'-=-=-=-=-=-=-',k)
            label[k.split('/')[-1].split('_')[0]]+=1
        print(label,'label')
        lab = check_max(label)
        print('file %s max label is:%d'%(j,lab))
        total = len(glob.glob(j+'/*'))
        same = 0
        # all_correct = 0
        # all_num = 0
        for k in glob.glob(j+'/*'):
            diff_0,diff_1,diff_2,diff_3,diff_4,diff_5,diff_6,diff_7 = count(int(k.split('/')[-1].split('_')[0]),diff_0,diff_1,diff_2,diff_3,diff_4,diff_5,diff_6,diff_7)
            all_0,all_1,all_2,all_3,all_4,all_5,all_6,all_7 = count(lab,all_0,all_1,all_2,all_3,all_4,all_5,all_6,all_7)
            if lab == int(k.split('/')[-1].split('_')[0]):
                count_right(k,lab,class_num)
                same +=1
                same_0,same_1,same_2,same_3,same_4,same_5,same_6,same_7 = count(int(lab),same_0,same_1,same_2,same_3,same_4,same_5,same_6,same_7)
                with open('log/%s/right_result_%s'%(d_name,class_num),'a',encoding='utf-8') as f_1:
                    f_1.write('the file is: %s predict is %d real is: %s\n'%(k.split('/')[-1],lab,k.split('/')[-1].split('_')[0]))
                with open('log/%s/all_result_%s'%(d_name,class_num),'a',encoding='utf-8') as f_2:
                    f_2.write('%s\t%d\n'%(k.split('/')[-1],lab))
            else:
                # count_error(k,lab,class_num)
                with open('log/%s/result_%s'%(d_name,class_num),'a',encoding='utf-8') as f_:
                    f_.write('the file is: %s predict is %d real is: %s\n'%(k.split('/')[-1],lab,k.split('/')[-1].split('_')[0]))
                with open('log/%s/all_result_%s'%(d_name,class_num),'a',encoding='utf-8') as f_2:
                    f_2.write('%s\t%d\n'%(k.split('/')[-1],lab))
                num[k.split('/')[-1].split('_')[0]]+=1
        num[str(lab)] = same
        all_correct+=same
        all_num += total
        if total == 0:
            total = 1
        for ks,vs in num.items():
            with open('log/%s/check_result_%s'%(d_name,class_num),'a',encoding='utf-8') as f:
                f.write('the file is: %s predict is %s total is: %s real is: %s num is %s acc is %f\n'%(j,str(lab),str(total),str(ks),str(vs),same/total))
    # print([[0,same_0,diff_0],[1,same_1,diff_1],[2,same_2,diff_2],[3,same_3,diff_3]])
    # exit()
a = []
b = []
# c = [all_0,all_1,all_2,all_3,all_4,all_5,all_6,all_7]
# for i in range(0,8):
    # if c[i] == 0:
        # c[i] = 1
# print(all_0,all_1,all_2,all_3,all_4,all_5,all_6,all_7)
with open('log/%s/check_result_%s'%(d_name,class_num),'a',encoding='utf-8') as f:
    f.write('The overrall correct is: %f\n'%(all_correct/all_num))
    # exit()
    print([[0,same_0,diff_0],[1,same_1,diff_1],[2,same_2,diff_2],[3,same_3,diff_3]])
    for i in [[0,same_0,diff_0,all_0],[1,same_1,diff_1,all_1],[2,same_2,diff_2,all_2],[3,same_3,diff_3,all_3],[4,same_4,diff_4,all_4],[5,same_5,diff_5,all_5],[6,same_6,diff_6,all_6],[7,same_7,diff_7,all_7]]:
        try:
            a.append(i[1]/i[2])
        except:
            i[2]=1
            a.append(0.0)
        # a.append(i[1]/i[2])
        try:
            b.append(i[1]/i[3])
        except:
            i[3]=1
            b.append(0.0)
        f.write('The lab %d all is: %d ,correct is: %d\n'%(i[0],i[2],i[1]))
        f.write('The lab %d recall is: %f\n'%(i[0],i[1]/i[2]))
        f.write('The lab %d acc is: %f\n'%(i[0],i[1]/i[3]))
with open('log/%s/check_result_%s'%(d_name,class_num),'a',encoding='utf-8') as f:
    f.write('The dropping is: %f\n'%(round((nums-all_num)/nums,3)))
print(all_correct,all_num,round((nums-all_num)/nums,3),nums,nums1)

train_xsl_=[['label_0','label_1','label_2','label_3','label_4','label_5','label_6','label_7'],[diff_0,diff_1,diff_2,diff_3,diff_4,diff_5,diff_6,diff_7],[same_0,same_1,same_2,same_3,same_4,same_5,same_6,same_7],a,b]
train_data_ = {'name':train_xsl_[0],'all':train_xsl_[1],'right':train_xsl_[2],'recall':train_xsl_[3],'acc':train_xsl_[4]}
df = pd.DataFrame(train_data_)
df.to_excel('log/%s/result_%s.xlsx'%(d_name,class_num),index = False)
