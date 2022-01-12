import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image


# 特徴抽出器
def load_feature_extractor():
    vgg16 = VGG16(weights='imagenet', include_top=True)
    feature_extractor = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)
    return feature_extractor


# センタークロップ
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


# 前処理
def preprocess_vgg16(pil_img):
    # リサイズ
    long_edge = max(pil_img.size)
    resized_img = crop_center(pil_img, long_edge, long_edge).resize((224, 224), Image.LANCZOS)
    # 変換
    input_array = [img_to_array(resized_img)]
    array_preprocessed = preprocess_input(np.array(input_array))
    return array_preprocessed


# CNN特徴量を抽出
def extract_cnn_feature(model, inputs):
    feature = model.predict(inputs).reshape([inputs.shape[0], -1])
    return feature
