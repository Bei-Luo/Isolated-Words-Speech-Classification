import base64
import json

import flask
import torch
from flask import request, send_from_directory, render_template

from model.conv import vgg11_bn
from preprocess import mel_spec

words = ['数字', '语音', '语言', '识别', '中国', '总工', '北京', '背景', '上海', '商行']

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = vgg11_bn()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint = torch.load('./save/save.ptr')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
    model.eval()

    app = flask.Flask(__name__, static_folder='static')


    @app.route('/', methods=['POST', 'GET'])
    def home():
        return render_template('recorder.html')


    @app.route('/infer', methods=['GET', 'POST'])
    def infer():
        if request.method == 'GET':
            data = request.args.get('data')
            name = request.args.get('name')
        else:
            data = request.form.get('data')
            name = request.form.get('name')
        # print(data)
        data = data[22:]
        base64_data = data
        ori_image_data = base64.b64decode(base64_data)
        fout = open('./test_data/' + name, 'wb')
        fout.write(ori_image_data)
        fout.close()
        X = mel_spec('./test_data/' + name)
        X = X.reshape(1, X.shape[0], -1)
        X = torch.from_numpy(X).type(torch.float)
        X = X[None, :, :, :]
        X = X.to(device)
        y = model(X)
        inx = int(y.argmax(dim=1))
        if inx > len(words):
            word = '识别错误'
        else:
            word = words[inx]
        ret = {
            'word': word
        }
        return json.dumps(ret)


    @app.route('/js/<path:path>')
    def send_js(path):
        return send_from_directory('interface/js', path)


    app.run(host="127.0.0.1", port=8800)
