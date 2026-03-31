import UMAP
import matplotlib.pyplot as plt
import torch
import numpy as np

def save_data(label,data,name):
    with open(name,"w",encoding="utf-8")as f:
        for pa,da in zip(label,data):
            to_write="<=>".join([str(x)for x in da])
            f.write("%s\t%s\n"%(pa,to_write))

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
        
    data = []
    label = []
    for line in lines:
        la = line.split('\t')[0].split('/')[-1]
        label.append(la)
        
        da = line.split('\t')[-1]
        da = [float(i) for i in da.split('<=>')]
        data.append(da)

    return data,label


if __name__=="__main__":
    encode_path = '/home/daiyao/zsw/新umap/zsw/sample/cluster/data/encode_convnext.txt'
    method = 'new_UMAP_13'
    #添加降维维度
    k_lst=[10,20,50,100,200,300,500,900,1200]
    data,label = read_txt(encode_path)

    data_ = np.array(data)
    data_ = torch.tensor(data_, dtype=torch.float)
    print(data_.shape)
    
    rec_dict=UMAP.main(data_,k_lst)
    # print(rec_dict)

    for k,data in rec_dict.items():
        name="../dimension_result/%s_%s"%(method,k)
        encode=data["encode"]
        print(encode.shape)
        save_data(label,encode,name)