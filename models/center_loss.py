import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision.transforms as transforms

# configuration
NUM_CLASSES = 3
IMG_SIZE = 320
MODEL_PATH = './weight/cln.pth'

# center loss network
class CenterLossNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        base_model = resnet18(pretrained=False)
        self.features = nn.Sequential(
            *[layer for layer in base_model.children()][:-1],
            nn.Conv2d(512, 512, 1, bias=False),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        self.centers = nn.Parameter(torch.zeros(num_classes, 512),  requires_grad=False)
        self.num_classes = num_classes
        self.feature_ = None
        
    def forward(self, x):
        feature = self.features(x)
        feature = feature.view(-1, 512)
        self.feature_ = feature
        return self.fc(feature)


# clnの読み込み
def load_clnet():
    cln = CenterLossNetwork(num_classes=NUM_CLASSES)
    cln.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    cln.eval()
    return cln
    


# clnの前処理
def preprocess_cln(img):
    transformer = transforms.Compose([
        transforms.Resize(size=IMG_SIZE),
        transforms.ToTensor()
    ])

    preprocessed_img = transformer(img)
    input_tensor = torch.unsqueeze(preprocessed_img, 0) # バッチ次元の追加
    return input_tensor

# スコアを取得
def predic_cln(model, inputs):
    class_names = ['overripe', 'ripe', 'unripe']
    outputs = model(inputs)
    batch_probs = F.softmax(outputs, dim=1)
    
    result_dic = {}

    for i, class_name in enumerate(class_names):
        prob = round(float(batch_probs[0][i]), 4) # バッチ次元は0を指定, 少数4位で丸める
        result_dic[class_name] = prob

    return result_dic