import json
from pathlib import Path
import torch
from torchvision import models, transforms
from torch.nn import functional as F
from PIL import Image


# vggモデルのロード
def load_vgg():
    vgg = models.vgg16(pretrained=True)
    vgg.eval()
    return vgg


# vggの前処理
def preprocess_vgg(img):
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    preprocessed_img = transformer(img) # 前処理適用
    input_tensor = torch.unsqueeze(preprocessed_img, 0) # バッチ次元追加
    return input_tensor

# image_netのクラスをリスト化
def get_classes():
    with open("./imagenet_class_index.json") as f:
        data = json.load(f)
        class_names = [v[1] for v in data.values()]

    return class_names


#top3のスコアを辞書化
def predict_vgg(model, inputs):
    class_names = get_classes()
    outputs = model(inputs)
    batch_probs = F.softmax(outputs, dim=1)
    batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)
    
    result_dic = {}
    for probs, indices in zip(batch_probs, batch_indices):
        for k in range(3):
            result_dic[class_names[indices[k]]] = float(probs[k])
    
    return result_dic

