import PIL.Image as I
import torch
import os
import glob
from torchvision import transforms, models
from pytorch_pretrained_vit import ViT
import numpy as np
from tqdm import tqdm
import timm

def makedata(data):
    new = []
    for i in data:
        new.append(i)
    return new

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 提示选择模型
model_select = input("选择编码模型:1 - ViT, 2 - ConvNeXt, 3 - ResNet50, 4 - AlexNet, 5 - EfficientNet:")

if model_select == '1':
    model = ViT('B_16_imagenet1k', pretrained=True)
    img_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    filename = 'encode_vit.txt'
elif model_select == '2':
    model = timm.create_model('convnext_xlarge_in22k', pretrained=True, num_classes=0)
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    filename = 'encode_convnext.txt'
elif model_select == '3':
    model = timm.create_model('resnet50', pretrained=True, num_classes=0)
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    filename = 'encode_resnet50.txt'
elif model_select == '4':
    model = models.alexnet(pretrained=True)
    # 移除分类器的最后一层以获取特征
    model.classifier = model.classifier[:-1]
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    filename = 'encode_alexnet.txt'
elif model_select == '5':
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)  # 可以根据需要更改EfficientNet版本
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据所选的EfficientNet版本调整大小
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    filename = 'encode_efficientnet.txt'
else:
    print("无效的选择!")
    exit()

model = model.to(device)
model.eval()

# 数据集地址
data_path = r"/home/daiyao/zsw/新umap/polar_img/"

# 编码和保存
imgs = []

for image_path in tqdm(glob.glob(data_path + '/*')):
    img = I.open(image_path).convert('RGB')
    img = img_transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        img = img.to(device)
        outputs = model(img)
        outputs = outputs.cpu().numpy().flatten()
        np.set_printoptions(suppress=True)

        imgs.append(outputs)
        datas = makedata(outputs)

        # 根据选择的模型保存到不同文件
        with open("/share/shiliang/yingxiaolei/umap/" + filename, 'a') as f:
            f.write('%s\t%s\n' % (image_path, '<=>'.join([str(x) for x in datas])))