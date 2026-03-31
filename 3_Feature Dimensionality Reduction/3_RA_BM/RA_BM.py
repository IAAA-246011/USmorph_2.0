import numpy as np
from sklearn.random_projection import SparseRandomProjection


def save_data(labels, data, name):
    """保存降维后的特征数据，保留标签用于后续聚类结果分析"""
    with open(name, "w", encoding="utf-8") as f:
        for label, features in zip(labels, data):
            to_write = '<=>'.join([str(x) for x in features])
            f.write(f"{label}\t{to_write}\n")
    print(f"降维结果已保存至: {name}")


def read_txt(txt_path):
    """读取原始特征向量数据"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    labels = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        label, feature_str = parts
        try:
            features = [float(x) for x in feature_str.split('<=>')]
            data.append(features)
            labels.append(label)
        except ValueError:
            continue

    return np.array(data), labels


# -----------------------------
# 方法 1: 随机投影 (Sparse Random Projection)
# -----------------------------
def random_projection_reduction(features, n_components=100, random_state=42):
    """
    使用稀疏随机投影进行降维
    计算简单、速度快，适合大规模数据
    """
    print(f"正在执行 Sparse Random Projection，目标维度: {n_components}")
    transformer = SparseRandomProjection(n_components=n_components, random_state=random_state)
    return transformer.fit_transform(features)


# -----------------------------
# 方法 2: 特征分块取均值 (Feature Binning + Mean)
# -----------------------------
def binning_mean_reduction(features, n_components=100):
    """
    将高维特征分块，每块取均值，实现简单降维
    无需模型、无需训练，纯手工操作
    """
    print(f"正在执行 Feature Binning + Mean Pooling，目标维度: {n_components}")
    n_samples, n_features = features.shape

    # 调整 n_components 以确保能整除（或接近）
    bin_size = n_features // n_components
    if bin_size == 0:
        raise ValueError("n_components 过大，无法分块")

    # 截断到可整除长度
    valid_length = bin_size * n_components
    if valid_length > n_features:
        valid_length = n_features - (n_features % bin_size)  # 向下对齐
    features_trunc = features[:, :valid_length]

    # 重塑并取均值
    features_reshaped = features_trunc.reshape(n_samples, n_components, bin_size)
    return features_reshaped.mean(axis=2)  # shape: (n_samples, n_components)


if __name__ == "__main__":
    # 配置参数
    encode_path = r'/060111/con_umap/CLASS/encode_alexnet.txt'
    output_dir = r'/060111/con_umap/CLASS/'
    n_components = 100  # 降维目标维度

    # 支持的方法（可选 'random_projection' 或 'binning_mean'）
    methods = ['binning_mean']  # 可改为 ['binning_mean'] 或两者都试

    # 读取原始特征
    features, labels = read_txt(encode_path)
    print(f"原始特征形状: {features.shape}")

    # 遍历不同方法
    for method in methods:
        try:
            if method == 'random_projection':
                reduced_features = random_projection_reduction(features, n_components=n_components)
            elif method == 'binning_mean':
                reduced_features = binning_mean_reduction(features, n_components=n_components)
            else:
                print(f"不支持的方法: {method}")
                continue

            print(f"{method} 降维后特征形状: {reduced_features.shape}")

            # 保存结果
            output_filename = f"{output_dir}alexnet_{method}_components_{n_components}.txt"
            save_data(labels, reduced_features, output_filename)

        except Exception as e:
            print(f"{method} 降维失败: {e}")