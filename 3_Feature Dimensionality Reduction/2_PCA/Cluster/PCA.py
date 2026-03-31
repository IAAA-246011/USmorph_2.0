# -*- coding: utf-8 -*-
# -*- coding: gbk -*-

from numpy import *
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import PIL.Image as I
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams.update({
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
})


def save_image(x):
    for ll, xx in enumerate(x):
        img = xx.reshape(32, 32)
        img_ = I.fromarray(int8(img))
        img_.save("%s.png" % ll)


def plot_100_image(x, name):
    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    for c in range(10):
        for r in range(10):
            ax[c, r].imshow(x[10 * c + r].reshape(32, 32).T, cmap='Greys_r')
            ax[c, r].set_xticks([])
            ax[c, r].set_yticks([])
    plt.savefig(name)
    plt.close()


def reduce_mean(X):
    X_reduce_mean = X - X.mean(axis=0)
    return X_reduce_mean


def sigma_matrix(X_reduce_mean):
    sigma = (X_reduce_mean.T @ X_reduce_mean) / X_reduce_mean.shape[0]
    return sigma


def usv(sigma):
    u, s, v = linalg.svd(sigma)
    return u, s, v


def project_data(X_reduce_mean, u, k):
    u_reduced = u[:, :k]
    z = dot(X_reduce_mean, u_reduced)
    return z


def recover_data(z, u, k):
    u_reduced = u[:, :k]
    X_recover = dot(z, u_reduced.T)
    return X_recover


def calculate_ratios(lst):
    ratios = []
    for i in range(1, len(lst)):
        ratio = lst[i] / lst[i - 1]
        ratios.append(ratio)
    return ratios


def Do(x, k_lst):
    x_reduce_mean = reduce_mean(x)
    sigma = sigma_matrix(x_reduce_mean)
    u, s, v = usv(sigma)

    print(s)

    # 绘制奇异值
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    ax_s.plot(s, linewidth=2.0)
    ax_s.set_xlabel('dimension')
    ax_s.set_ylabel('singular value')
    ax_s.tick_params(axis='both', direction='in', length=6, width=1.0)
    fig_s.tight_layout()
    fig_s.savefig('s.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig_s)

    # 计算比率并转为 numpy 数组，避免布尔索引报错
    xx = np.array(calculate_ratios(s))
    dims = np.arange(1, len(xx) + 1)

    # 主图
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(dims, xx, linewidth=2.2)
    ax.set_xlabel('dimension')
    ax.set_ylabel('ratio')
    ax.axvline(2900, linestyle='--', linewidth=1.4)
    ax.set_xlim(0, len(xx))
    ax.tick_params(axis='both', direction='in', length=6, width=1.0)
    ax.grid(False)

    # inset 放大图：显示 2900 附近
    axins = inset_axes(
        ax,
        width="42%",
        height="42%",
        loc='lower left',
        bbox_to_anchor=(0.48, 0.10, 0.5, 0.5),
        bbox_transform=ax.transAxes,
        borderpad=1
    )

    axins.plot(dims, xx, linewidth=2.0)
    axins.axvline(2900, linestyle='--', linewidth=1.2)

    # 这里可按实际曲线微调范围
    x1, x2 = 2800, 3000
    mask = (dims >= x1) & (dims <= x2)

    axins.set_xlim(x1, x2)

    if np.any(mask):
        y_local = xx[mask]
        y_min = np.min(y_local)
        y_max = np.max(y_local)
        dy = (y_max - y_min) * 0.15 if y_max > y_min else 0.02
        axins.set_ylim(y_min - dy, y_max + dy)

    axins.tick_params(axis='both', direction='in', labelsize=9, length=4, width=0.8)
    axins.grid(False)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=1.0)

    fig.tight_layout()
    fig.savefig('s_ratio.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('s_ratio.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    ret_dict = {}
    for k in k_lst:
        z = project_data(x_reduce_mean, u, k)
        x_recover = recover_data(z, u, k)
        ret_dict[k] = {"recover": x_recover, "encode": z}

    return ret_dict


if __name__ == "__main__":
    faces_data = loadmat('/home/zhouchichun/Tensor/datas/tmp/face/ex7faces.mat')
    X = faces_data['X']
    print(X.shape)

    name = "raw.png"
    plot_100_image(X, name)

    k_lst = [100, 200, 300, 400, 500]
    x_recover_dict = Do(X, k_lst)

    for k, x_recover in x_recover_dict.items():
        name = "%s.png" % k
        recover = x_recover["recover"]
        encode = x_recover["encode"]
        plot_100_image(recover, name)