import numpy as np
import torch
import umap
import matplotlib.pyplot as plt

# 示例数据，假设你有一个 PyTorch 张量 encoding_tensor，形状为 (n_samples, n_features)
# 首先将数据转换为 NumPy 数组
def main(d,k):
    data_np = d.numpy()

    # 使用 umap-learn 进行降维
    ret_dict={}
    for i in k:
        reducer = umap.UMAP(n_components=i,n_neighbors=12)#min_dist=0.1
        
     
        embedding_np = reducer.fit_transform(data_np)
        ret_dict[i]={"encode":embedding_np}
    # 将降维后的结果转换为 PyTorch 张量
    return ret_dict
    

def mian2(d):
    data_np = d.numpy()
    reducer = umap.UMAP(n_components=2, n_neighbors=12, min_dist=0.1)
    umap_result = reducer.fit_transform(data_np)

    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels)  # labels 是数据点的标签
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("UMAP Visualization")
    plt.show()
if __name__=="__main__":
    main()
    

