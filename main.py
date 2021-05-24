from flask import Flask, request, jsonify
from PIL import Image
import io
from models.model import load_vgg, preprocess_vgg, predict_vgg
from models.center_loss import load_clnet, preprocess_cln, predic_cln

app = Flask(__name__)
#model = load_vgg()
model = load_clnet()


# 接続テスト用
@app.route('/')
def index():
    return 'Hello World!'

@app.route('/hello')
def test():
    return jsonify({'message': 'Hello world'})

# 画像予測用のpostメソッド
@app.route('/predict/vgg', methods=['POST'])
def predict():
    if request.files['image']:
        # 画像読み込み
        img = request.files['image'].read()
        img = Image.open(io.BytesIO(img)) # pillow形式
        # 前処理
        input_tensor = preprocess_vgg(img)
    
        result_json = jsonify(predict_vgg(model, input_tensor))
            # レスポンス
        return result_json
    else:
        return 'Cannot predict image.'

# cln予測用のメソッド
@app.route('/predict/cln', methods=['POST'])
def predict_cln():
    if request.files['image']:
        # 画像読み込み
        img = request.files['image'].read()
        img = Image.open(io.BytesIO(img)) # pillow形式
        # 前処理
        input_tensor = preprocess_cln(img)
    
        result_json = jsonify(predic_cln(model, input_tensor))
            # レスポンス
        return result_json
    else:
        return 'Cannot predict image.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)
