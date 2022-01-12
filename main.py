from flask import Flask, request, jsonify
from PIL import Image
import io
from models.master import extract_cnn_feature, load_feature_extractor, preprocess_vgg16
from models.center_loss import load_clnet, preprocess_cln, predic_cln
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np


app = Flask(__name__)

# その他
clnet_model = load_clnet() # center loss resnet18
vgg_extractor = load_feature_extractor() # 特徴抽出器

# 固定環境
kotei_scaler = pickle.load(open('./sklearn/kotei_scaler.sav', 'rb'))
kotei_model = pickle.load(open('./sklearn/kotei_nn.sav', 'rb'))
# 非固定環境
hikotei_scaler = pickle.load(open('./sklearn/hikotei_scaler.sav', 'rb'))
hikotei_model = pickle.load(open('./sklearn/hikotei_nn.sav', 'rb'))



# 接続テスト用
@app.route('/')
def index():
    return 'Hello World!'

@app.route('/hello')
def test():
    return jsonify({'message': 'Hello world'})


# cln予測用のメソッド
@app.route('/predict/cln', methods=['POST'])
def predict_cln():
    if request.files['image']:
        # 画像読み込み
        img = request.files['image'].read()
        img = Image.open(io.BytesIO(img)) # pillow形式
        # 前処理
        input_tensor = preprocess_cln(img)
    
        result_json = jsonify(predic_cln(clnet_model, input_tensor))
        # レスポンス
        return result_json
    else:
        return 'Cannot predict image.'


# 固定環境
@app.route('/predict/kotei', methods=['POST'])
def predict_kotei():
    if request.files['image']:
        # 画像読み込み
        img = request.files['image'].read()
        img = Image.open(io.BytesIO(img)) # pillow形式
        # 前処理
        input_tensor = preprocess_vgg16(img)
        # 特徴量取得
        cnn_feature = extract_cnn_feature(vgg_extractor, input_tensor)
        # 分類
        std_feature = kotei_scaler.transform(cnn_feature) # 標準化
        result = np.round(kotei_model.predict_proba(std_feature), 3) # 結果を少数3桁に変換
        # JSON変換
        result_dic = {'unripe': float(result[0][0]), 'ripe': float(result[0][1]), 'overripe': float(result[0][2])}
        result_json = jsonify(result_dic)
        # レスポンス
        return result_json
    else:
        return 'Cannot predict image.'


# 非固定環境
@app.route('/predict/hikotei', methods=['POST'])
def predict_hikotei():
    if request.files['image']:
        # 画像読み込み
        img = request.files['image'].read()
        img = Image.open(io.BytesIO(img)) # pillow形式
        # 前処理
        input_tensor = preprocess_vgg16(img)
        # 特徴量取得
        cnn_feature = extract_cnn_feature(vgg_extractor, input_tensor)
        # 分類
        std_feature = hikotei_scaler.transform(cnn_feature) # 標準化
        result = np.round(hikotei_model.predict_proba(std_feature), 3) # 結果を少数3桁に変換
        # JSON変換
        result_dic = {'unripe': float(result[0][0]), 'ripe': float(result[0][1]), 'overripe': float(result[0][2])}
        result_json = jsonify(result_dic)
        # レスポンス
        return result_json
    else:
        return 'Cannot predict image.'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)
