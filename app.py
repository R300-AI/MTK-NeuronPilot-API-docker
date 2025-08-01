import subprocess, json, time
from flask import (Flask, redirect, request, send_from_directory, jsonify, Response)
from werkzeug.utils import secure_filename
from utils.file import verify_uploded_file
from utils.converter import convert_pytorch_to_tflite
import torch
import os
import tensorflow as tf
import time
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

@app.route('/', methods=['GET', 'POST'])
def api_index():
    # 清理超過一天未更新的 users/* 目錄
    users_root = './users'
    now = time.time()
    expire_seconds = 24 * 60 * 60
    if os.path.exists(users_root):
        for user_dir in os.listdir(users_root):
            user_path = os.path.join(users_root, user_dir)
            if os.path.isdir(user_path):
                try:
                    mtime = os.path.getmtime(user_path)
                    if now - mtime > expire_seconds:
                        shutil.rmtree(user_path)
                        print(f"[CLEANUP] Removed expired user dir: {user_path}")
                except Exception as e:
                    print(f"[CLEANUP] Error checking/removing {user_path}: {e}")

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
            if action == 'convert_tflite':
                print('==> api_index: call download_dla')
                return download_dla()
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
    return Response(verify_uploded_file(
        filename, 
        save_path,
        user_id
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

@app.route('/download_dla', methods=['POST'])
def download_dla():
    print("==> download_dla API called")  # Debugging statement
    user_id = request.headers.get('X-User-ID')
    data = request.get_json()
    target_device = data.get('device').strip('.0')
    file_path = f'./users/{user_id}/export/{target_device}/model.dla'
    print(f"==> Looking for file: {file_path}")  # Debugging statement
    if not os.path.exists(file_path):
        print(f"==> File not found: {file_path}")  # Debugging statement
        return jsonify({"error": "Requested file not found"}), 404

    print(f"==> File found, sending: {file_path}")  # Debugging statement
    return send_from_directory(
        directory=os.path.dirname(file_path),
        path=os.path.basename(file_path),
        as_attachment=True,
        mimetype='application/octet-stream'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8086, debug=False)
