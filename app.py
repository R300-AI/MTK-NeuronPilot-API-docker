import subprocess, json, time
from flask import (Flask, redirect, request, send_from_directory, jsonify, Response)
from werkzeug.utils import secure_filename
from utils.file_namagement import allowed_file, random_seed
from utils.converter import convert_pytorch_to_tflite


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print("[upload_file] open index.html")
    with open('index.html', 'r') as file:
        html = file.read()
    print(f"[upload_file] request.method = {request.method}")
    if request.method == 'POST':
        print("[upload_file] POST received")
        if request.is_json:
            print("[upload_file] request.is_json = True")
            data = request.get_json()
            print(f"[upload_file] data = {data}")
            action = data.get('action', 'convert_tflite')
            print(f"[upload_file] action = {action}")
            if action == 'verify_pytorch':
                print("[upload_file] call handle_pytorch_verification")
                return handle_pytorch_verification(request)
        else:
            print("[upload_file] request.is_json = False")
            action = request.form.get('action', 'convert_tflite')
            print(f"[upload_file] form action = {action}")
            if action == 'convert_tflite':
                print("[upload_file] call handle_tflite_conversion")
                return handle_tflite_conversion(request, html)
    print("[upload_file] return html")
    return html

@app.route('/verify_pytorch', methods=['POST'])
def handle_pytorch_verification():
    print("[handle_pytorch_verification] IN /verify_pytorch")
    data = request.get_json()
    print(f"[handle_pytorch_verification] data = {data}")
    user_id = request.headers.get('X-User-ID')
    print(f"[handle_pytorch_verification] user_id = {user_id}")
    pytorch_code = data.get('pytorch_code', '')
    print(f"[handle_pytorch_verification] pytorch_code length = {len(pytorch_code)}")
    model_entrypoint = data.get('model_entrypoint', 'SimpleModel')
    print(f"[handle_pytorch_verification] model_entrypoint = {model_entrypoint}")
    input_shape = data.get('input_shape', '(1, 10)')
    print(f"[handle_pytorch_verification] input_shape = {input_shape}")
    print("[handle_pytorch_verification] call convert_pytorch_to_tflite")
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

def handle_tflite_conversion(request, html):
    genio_device = request.form.get('genio_device')
    device = request.form.get('device')
    
    print(f"📝 Genio Device: {genio_device}")
    print(f"📝 Device: {device}")
    
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
        return redirect(request.url)
    if file and allowed_file(file.filename):
        tflite_name = secure_filename(file.filename)
        seed = random_seed()
        print(f'📝 Generate upload instance: {seed}')
        print(f'📝 Processing for {genio_device.upper()} with {device}')
        os.makedirs(f"{app.config['UPLOAD_FOLDER']}/{seed}", exist_ok=True)
        saved_path = os.path.join(f"{app.config['UPLOAD_FOLDER']}/{seed}", tflite_name)
        file.save(saved_path)
        cmd = f"./neuronpilot-6.0.5/neuron_sdk/host/bin/ncc-tflite --arch={device} --relax-fp32 {saved_path}"
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
    app.run(host='0.0.0.0', port=8092, debug=False)
