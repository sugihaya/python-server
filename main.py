from flask import Flask, request, jsonify
from PIL import Image
import io
from model import load_vgg, preprocess_vgg, predict_vgg

app = Flask(__name__)
model = load_vgg()

# 接続テスト用
@app.route('/hello')
def test():
    return jsonify({'message': 'Hello world'})

# 画像予測用のpostメソッド
@app.route('/predict', methods=['POST'])
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


if __name__ == '__main__':
    app.run(debug=False, port=5000)
