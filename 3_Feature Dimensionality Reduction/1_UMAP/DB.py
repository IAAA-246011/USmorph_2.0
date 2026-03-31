import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score  # 使用 DB 指数
from skopt import gp_minimize
from skopt.space import Integer
import csv
import os
import random

# 固定随机种子以确保结果可重复性
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# 示例数据处理函数
def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        # 提取特征向量
        da = line.split('\t')[-1]
        da = [float(i) for i in da.split('<=>')]
        data.append(da)

    return data


# UMAP降维函数
def umap_reduce(data_np, n_components):
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=100,
        metric='cosine',       #'euclidean',
        min_dist=0.1,
        random_state=42
    )
    embedding_np = reducer.fit_transform(data_np)
    return embedding_np


# 聚类并计算 DB 指数的函数
def cluster_and_score(embedding, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embedding)
    db_index = davies_bouldin_score(embedding, kmeans.labels_)
    return kmeans.labels_, db_index  # 返回聚类标签和 DB 指数


# 目标函数用于贝叶斯优化
def objective_function(params):
    n_components = params[0]  # 提取第一个参数
    print(f"Trying n_components={n_components}")  # 打印当前尝试的参数

    # 使用当前的 n_components 进行 UMAP 降维
    embedding = umap_reduce(data_np, int(n_components))

    # 使用 KMeans 进行聚类并计算 DB 指数
    _, db_index = cluster_and_score(embedding, n_clusters=10)  # 假设簇数为 10
    scores[int(n_components)] = db_index  # 记录每个 n_components 的 DB 指数

    print(f"DB Index for n_components={n_components}: {db_index}")
    return db_index  # 直接返回 DB 指数（因为越小越好）


# 主运行代码
if __name__ == "__main__":
    encode_path = r"/share/shiliang/yingxiaolei/umap/encode_result_100"
    method = 'new_UMAP_13'
    data = read_txt(encode_path)
    data_np = np.array(data)

    # 初始化存储 DB 指数的字典
    scores = {}

    # 使用贝叶斯优化选择最佳的 n_components
    search_space = Integer(low=50, high=500, name='n_components')
    result = gp_minimize(objective_function, dimensions=[search_space], n_calls=30, random_state=42)

    best_n_components = result.x[0]
    print(f"Best number of components found: {best_n_components}")

    # 将 DB 指数保存到 CSV 文件
    csv_path = 'db_scores.csv'
    try:
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['n_components', 'db_score'])
            for n_comp, db_score in scores.items():
                writer.writerow([n_comp, db_score])
        print(f"DB scores saved to {os.path.abspath(csv_path)}")
    except Exception as e:
        print(f"Failed to write DB scores to CSV: {e}")

    # 使用最佳的 n_components 进行最终的 UMAP 降维
    final_embedding = umap_reduce(data_np, best_n_components)
    print(f"Final embedding shape: {final_embedding.shape}")

    # 使用 KMeans 进行聚类
    n_clusters = 10  # 假设我们选择 10 个簇
    cluster_labels, _ = cluster_and_score(final_embedding, n_clusters)

    # 可视化最终的结果（如果 n_components >= 2）
    if best_n_components >= 2:
        if final_embedding.shape[1] < 2:
            raise ValueError(
                "Final embedding has fewer than 2 dimensions. Visualization requires at least 2 dimensions.")

        # 绘制二维散点图
        plt.figure(figsize=(12, 10))  # 调整图像大小
        scatter = plt.scatter(
            final_embedding[:, 0],
            final_embedding[:, 1],
            c=cluster_labels,
            cmap='tab20',  # 使用 tab20 离散色图，适合分类数据
            alpha=0.8,  # 提高透明度以便更好观察重叠点
            s=50  # 调整点的大小
        )

        # 添加颜色条并调整标签
        cbar = plt.colorbar(scatter, label='Cluster Label')
        cbar.set_ticks(range(10))  # 设置颜色条刻度为 0 到 9
        cbar.set_ticklabels([f'Cluster {i}' for i in range(10)])  # 显示簇编号

        # 添加坐标轴标签和标题
        plt.xlabel("UMAP Dimension 1", fontsize=14)
        plt.ylabel("UMAP Dimension 2", fontsize=14)
        plt.title("UMAP Visualization with Best Components and Clustering", fontsize=16)

        # 保存图像文件
        plt.savefig('umap_visualization.png', dpi=300, bbox_inches='tight')  # 提高分辨率并裁剪多余空白
        plt.show()