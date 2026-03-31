0    python readimg_fits.py 将原始的fit文件读取为png格式的图像，并储存于fit_img/文件中

1    配置config.py   对模型参数进行配置，设置图像尺寸大小、卷积核大小参数（若效果不好可以调大尺寸，如[3, 3, 1, 16]改为[5, 5, 1, 16]）等，若其中的the_dim设置过大无法运行，可适当降低维度（如缩小一倍）。

2    python MAIN.py --train True   训练模型进行自编码，支持断点续接。

3    python MAIN.py --test True  cae_img/中得到输出的降噪图像。

4    python polar.py  将cae_img/的图像转为极坐标图像并输出至polar_img/中

