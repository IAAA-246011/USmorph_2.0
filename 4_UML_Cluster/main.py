import config as CF
import model as m
import glob
import os
import time

def load_data(file_):
    data=[]
    file_lst=[]
    for line in open(file_,"r",encoding="utf-8"):
        label,da =line.strip().split("\t")
        label=label.replace("\\","/")
        #print(da)
        da_=[]
        for x in da.split("<=>"):
            try:
                da_.append(float(x))
            except:
                print(x)
                da_.append(0.0)
        data.append(da_)
        #da=[float(x) for x in da.strip("<=>")]#(da)
        file_lst.append(label)
    return file_lst,data
st=time.time()
file_lst,data=load_data("encode_result")
#print(file_lst[:10])
print("sample number is %s, sample dim is %s"%(len(data),len(data[0])))
#exit()
class_num=CF.config["class_num"]
for model_type in CF.config["type"]: 
    model= m.model(class_num=class_num)
    model.build(model_type)
    model.run(data)
    model.show(file_lst)

print("聚类完成，一共用时:%s秒"%(time.time()-st))
