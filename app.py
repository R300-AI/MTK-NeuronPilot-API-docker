
import subprocess, json, time
from flask import (Flask, redirect, request, send_from_directory, jsonify, Response)
from werkzeug.utils import secure_filename
from utils.file import verify_and_convert_to_tflite
from utils.converter import convert_pytorch_to_tflite
import torch
import os
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

@app.route('/', methods=['GET', 'POST'])
def api_index():
    with open('index.html', 'r') as file:
        html = file.read()
        pytorch_version = torch.__version__
        tensorflow_version = tf.__version__
        html = html.replace(
            '<button type="button" class="tab-btn active" id="tab-pytorch" onclick="switchTab(\'pytorch\')">PyTorch</button>',
            f'<button type="button" class="tab-btn active" id="tab-pytorch" onclick="switchTab(\'pytorch\')">PyTorch={pytorch_version}</button>'
        )
        html = html.replace(
            '<button type="button" class="tab-btn" id="tab-tf" onclick="switchTab(\'tf\')">TensorFlow</button>',
            f'<button type="button" class="tab-btn" id="tab-tf" onclick="switchTab(\'tf\')">TensorFlow={tensorflow_version}</button>'
        )
    print('==> api_index: method =', request.method)
    if request.method == 'POST':
        if request.is_json:
            print('==> api_index: is_json')
            data = request.get_json()
            action = data.get('action', 'convert_tflite')
            if action == 'verify_model':
                print('==> api_index: call api_verify_model')
                return api_verify_model(request)
        else:
            print('==> api_index: not json')
            action = request.form.get('action', '')
            print(action)
            if action == 'upload_and_verify':
                print('==> api_index: call upload_and_verify')
                return upload_and_verify()
    return html

@app.route('/upload_and_verify', methods=['POST'])
def upload_and_verify():
    user_id = request.headers.get('X-User-ID')
    if 'upload_pretrained_file' not in request.files:
        def err():
            yield f"data: {{\"message\": \"❌ 沒有收到檔案 (upload_pretrained_file)\", \"error\": true, \"final\": true}}\n\n"
        return Response(err(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'Access-Control-Allow-Origin': '*'})
    file = request.files['upload_pretrained_file']
    filename = secure_filename(file.filename)
    save_dir = f'./users/{user_id}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    file.save(save_path)
    return Response(verify_and_convert_to_tflite(
        filename, 
        save_path
        ), 
        mimetype='text/event-stream', 
        headers={'Cache-Control': 'no-cache',
                 'Connection': 'keep-alive',
                 'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/verify_model', methods=['POST'])
def api_verify_model():
    data = request.get_json()
    user_id = request.headers.get('X-User-ID')
    pytorch_code = data.get('pytorch_code', '')
    tf_code = data.get('tf_code', '')
    model_entrypoint = data.get('model_entrypoint', 'SimpleModel')
    input_shape = data.get('input_shape', '(1, 10)')

    return Response(convert_pytorch_to_tflite(
        user_id=user_id,
        pytorch_code=pytorch_code,
        model_entrypoint=model_entrypoint,
        input_shape=input_shape
    ),
    mimetype='text/event-stream',
    headers={'Cache-Control': 'no-cache',
             'Connection': 'keep-alive',
             'Access-Control-Allow-Origin': '*'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083, debug=False)
