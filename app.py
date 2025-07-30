import subprocess, json, time
from flask import (Flask, redirect, request, send_from_directory, jsonify, Response)
from werkzeug.utils import secure_filename
from utils.file_namagement import allowed_file, random_seed
from utils.converter import convert_pytorch_to_tflite
import torch
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
    
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            action = data.get('action', 'convert_tflite')
            if action == 'verify_model':
                return api_verify_model(request)
        else:
            action = request.form.get('action', 'convert_tflite')
            if action == 'convert_model':
                return api_convert_model(request, html)
    return html

@app.route('/verify_model', methods=['POST'])
def api_verify_model():
    data = request.get_json()
    user_id = request.headers.get('X-User-ID')
    pytorch_code = data.get('pytorch_code', '')
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

def api_convert_model(request, html):
    genio_device = request.form.get('genio_device')
    device = request.form.get('device')
    
    
    device_mapping = {
        'genio510': ['mdla3.0', 'vpu'],
        'genio700': ['mdla3.0', 'vpu'],
        'genio1200': ['mdla2.0', 'vpu']
    }
    
    if not genio_device or genio_device not in device_mapping:
        html = html.replace('<h1>🔄 TensorFlow Lite Model Upload</h1>', 
                           '<h1>❌ Invalid Genio device selection.</h1>')
        return html
    
    if not device or device not in device_mapping[genio_device]:
        html = html.replace('<h1>🔄 TensorFlow Lite Model Upload</h1>', 
                           f'<h1>❌ Device {device} not supported for {genio_device.upper()}.</h1>')
        return html
    
    if 'file' not in request.files:
        html = html.replace('<h1>🔄 TensorFlow Lite Model Upload</h1>', 
                           '<h1>❌ .tflite file not found.</h1>')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        html = html.replace('<h1>🔄 TensorFlow Lite Model Upload</h1>', 
                           '<h1>❌ .tflite file not found.</h1>')
        tflite_name = secure_filename(file.filename)
        seed = random_seed()
        print(f'📝 Generate upload instance: {seed}')
        print(f'📝 Processing for {genio_device.upper()} with {device}')
        os.makedirs(f"{app.config['UPLOAD_FOLDER']}/{seed}", exist_ok=True)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"📝 Return code: {result.returncode}")
        print(f"📝 Output: {result.stdout}")
        print(f"📝 Error: {result.stderr}")
        if int(result.returncode) == 0:
            dla_name = tflite_name.rstrip('.tflite') + '.dla'
            response = send_from_directory(f"{app.config['UPLOAD_FOLDER']}/{seed}", dla_name)
            device_name = device.replace('.', '_')
            genio_suffix = genio_device.replace('genio', '')
            response.headers['name'] = dla_name.replace('.dla', f'_{genio_suffix}_{device_name}.dla')
            return response
        else:
            saved_path = os.path.join(f"{app.config['UPLOAD_FOLDER']}/{seed}", "error_message.txt")
            with open(saved_path, 'w') as f:
                f.write(f"❌ Genio Device: {genio_device}\n")
                f.write(f"❌ Target Device: {device}\n")
                f.write("❌ Error Output:\n")
                f.write(result.stdout + '\n' + result.stderr)
            response = send_from_directory(f"{app.config['UPLOAD_FOLDER']}/{seed}", "error_message.txt")
            response.headers['name'] = "error_message.txt"
            return response
    return html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083, debug=False)
