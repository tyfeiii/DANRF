import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Demo')

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--model', type=str, default='resnet50') # only support 'resnet50' right now
parser.add_argument('--part', type=str, default='sum')  # only support 'sum'  right now
parser.add_argument('--img_path', type=str, default='./test_data/')

args = parser.parse_args()
print(args)
mode = args.model
part = args.part
device = args.device
img_path = args.img_path

if device != 'cpu' and torch.cuda.is_available():
    device = "cuda"
else:
    device = 'cpu'


if mode == 'resnet50':
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(2048, 1, bias=True)
else:
    print("Please input right model name")
    os._exit(1)


model.to(device)
model.load_state_dict(torch.load('./checkpoint/' + mode + '_' + part))

model.eval()
loss_sum = 0
pred_lst = []
gt_lst = []

print(device)


img_size = (672, 672)
img_list = os.listdir(img_path)
img_list = img_list

print(img_list)
with torch.no_grad():
    for img_name in img_list:
        reading_path = img_path + img_name
        # print(reading_path)
        img = cv2.imread(reading_path)
        img = cv2.resize(img, img_size)
        x = np.array([img])
        x = x.transpose(0, 3, 1, 2) #trans (1,672,672,3) to (1,3,672,672)
        x = torch.from_numpy(x)
        x = x.type('torch.FloatTensor')
        x.to(device)
        xb = x.float()
        pred = model(xb)
        pred = torch.squeeze(pred)
        print(img_name, ': ', float(np.array(pred)))


