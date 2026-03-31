import config as CF
import model as m
import glob
import os, shutil, re
import time
import PCA
import numpy as np

def load_data(file_):
    data = []
    file_lst = []
    labels = []
    
    for line in open(file_, "r", encoding="utf-8"):
        try:
            na, da, label = line.strip().split("\t")
            file_lst.append(na)
            ss = da
            label = int(label)
            da_ = [float(x) if x else 0.0 for x in ss.split("<=>")]
            data.append(da_)
            labels.append(label)
        except Exception as e:
            print(f"Error processing line: {line.strip()}")
            print(e)
            
    print(len(labels), 'label==name', len(file_lst))
    return file_lst, data

def save_data(path, data, name):
    with open(name, "w", encoding="utf-8") as f:
        for pa, da in zip(path, data):
            to_write = "<=>".join([str(x) for x in da])
            lab = pa.split('/')[-1].split('_')[0]
            f.write("%s\t%s\t%s\n" % (pa, to_write, lab))

# 主程序
st = time.time()
path = 'data'  # 您可以在这里指定路径
file_lst, data = load_data("encode_result.txt")  # 直接加载现有的encode_result文件
method = 'PCA'
data_img = np.array(data)
k_lst = [800]
rec_dict = PCA.Do(data_img, k_lst)

if not os.path.exists('result_pca/'):
    os.makedirs('result_pca/', exist_ok=True)

for k, datas in rec_dict.items():
    name = "result_pca/%s_%s" % (method, k)
    encode = datas["encode"]
    recover = datas["recover"]
    save_data(file_lst, encode, name)

print("sample number is %s, sample dim is %s" % (len(data), len(data[0])))
m_name = input('please certain method：(pca/raw)')
class_num = int(input('请输入数据类别总数：'))

if m_name == 'pca':
    d_name = input('please print the dim of out:')
    file_lst, data = load_data("result_pca/PCA_%s" % (d_name))
    if os.path.exists('result/'):
        shutil.rmtree('result/')
    for model_type in CF.config["type"]:
        model = m.model(class_num=class_num)
        model.build(model_type)
        model.run(data)
        model.show(file_lst)
else:
    if os.path.exists('result/'):
        shutil.rmtree('result/')
    for model_type in CF.config["type"]:
        model = m.model(class_num=class_num)
        model.build(model_type)
        model.run(data)
        model.show(file_lst)

print("聚类完成，一共用时:%s秒" % (time.time() - st))
