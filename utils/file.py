import datetime, random
from .converter import onnx_to_tflite

def random_seed():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_number = random.randint(1000, 9999)
    return f"{timestamp}_{random_number}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'tflite'}

def verify_and_convert_to_tflite(filename, save_path):
    allowed_ext = {"onnx", "tflite"}
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in allowed_ext:
        yield f'data: {{"message": "[error] Only .onnx or .tflite files are supported (received: .{ext})", "error": true}}\n\n'
        return
    yield f'data: {{"message": "[info] File uploaded: {filename}"}}\n\n'
    if ext == "onnx":
        yield f'data: {{"message": "[info] Starting ONNX to TFLite conversion..."}}\n\n'
        try:
            for msg in onnx_to_tflite(filename, save_path):
                yield msg
            yield f'data: {{"message": "[success] Conversion completed. TFLite file generated."}}\n\n'
        except Exception as e:
            yield f'data: {{"message": "[error] Conversion failed: {str(e)}", "error": true}}\n\n'
    elif ext == "tflite":
        yield f'data: {{"message": "[info] TFLite file detected. No conversion needed."}}\n\n'
    

