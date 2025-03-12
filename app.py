import os, datetime, random, subprocess
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

def random_seed():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_number = random.randint(1000, 9999)
    return f"{timestamp}_{random_number}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'tflite'}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    html = """ 
    <!doctype html>
    <title>NeuronPilot Converter</title>
    <h1>Upload your .tflite model</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """
    if request.method == 'POST':
        print(request.url)
        if 'file' not in request.files:
            html.replace('<h1>Upload your .tflite model</h1>', '<h1>.tflite file not found.</h1>')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            html.replace('<h1>Upload your .tflite model</h1>', '<h1>.tflite file not found.</h1>')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            seed = random_seed()
            print(f'Generate a upload instance:{seed}')
            os.makedirs(f"{app.config['UPLOAD_FOLDER']}/{seed}", exist_ok=True)
            file_path = os.path.join(f"{app.config['UPLOAD_FOLDER']}/{seed}", filename)
            file.save(file_path)

            result = subprocess.run(f"./neuronpilot-6.0.5/neuron_sdk/host/bin/ncc-tflite --arch=mdla3.0 --relax-fp32 {file_path}", shell=True, capture_output=True, text=True)
            print(f"Return code: {result.returncode}")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")

            return send_from_directory(f"{app.config['UPLOAD_FOLDER']}/{seed}", filename)
    return html

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=80)



