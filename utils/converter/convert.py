
import os
import subprocess
import onnx
import numpy as np
import tensorflow as tf

def shape_match(a, b):
    """Check if two shapes match, allowing for NCHW <-> NHWC conversion."""
    if tuple(a) == tuple(b):
        return True
    if len(a) == 4 and len(b) == 4:
        # Allow (N, C, H, W) <-> (N, H, W, C)
        if a[0] == b[0] and a[1:] == b[-1:] + b[1:3]:
            return True
        if a[0] == b[0] and a[1:4] == b[1:4][::-1]:
            return True
        if a[0] == b[0] and a[1:] == b[1:][::-1]:
            return True
        if a[0] == b[0] and a[1:] == [b[3], b[1], b[2]]:
            return True
        if a[0] == b[0] and a[1:4] == [b[2], b[3], b[1]]:
            return True
        if a[0] == b[0] and a[1] == b[3] and a[2] == b[1] and a[3] == b[2]:
            return True
    return False

def tflite_to_mdla3(tflite_path):
    """Convert a TFLite file to DLA format using ncc-tflite. Returns the DLA file path."""
    try:
        device = 'mdla3.0'
        output_dir = os.path.dirname(tflite_path)
        dla_name = os.path.basename(tflite_path).replace('.tflite', '.dla')
        dla_path = os.path.join(output_dir, dla_name)
        ncc_bin = './neuronpilot-6.0.5/neuron_sdk/host/bin/ncc-tflite'
        cmd = [ncc_bin, f'--arch={device}', '--relax-fp32', tflite_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ncc-tflite failed: {result.stdout}\n{result.stderr}")
        if not os.path.exists(dla_path):
            raise RuntimeError(f"DLA 檔案未產生於 {dla_path}")
        print(f"[dla] 產生成功: {dla_path}")
        return dla_path
    except Exception as e:
        raise RuntimeError(f"TFLite to DLA conversion failed: {e}")

def tflite_to_mdla2(tflite_path):
    """Convert a TFLite file to DLA format using ncc-tflite. Returns the DLA file path."""
    try:
        device = 'mdla2.0'
        output_dir = os.path.dirname(tflite_path)
        dla_name = os.path.basename(tflite_path).replace('.tflite', '.dla')
        dla_path = os.path.join(output_dir, dla_name)
        ncc_bin = './neuronpilot-6.0.5/neuron_sdk/host/bin/ncc-tflite'
        cmd = [ncc_bin, f'--arch={device}', '--relax-fp32', tflite_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ncc-tflite failed: {result.stdout}\n{result.stderr}")
        if not os.path.exists(dla_path):
            raise RuntimeError(f"DLA 檔案未產生於 {dla_path}")
        print(f"[dla] 產生成功: {dla_path}")
        return dla_path
    except Exception as e:
        raise RuntimeError(f"TFLite to DLA conversion failed: {e}")
    

def tflite_to_vpu(tflite_path):
    """Convert a TFLite file to DLA format using ncc-tflite. Returns the DLA file path."""
    try:
        device = 'vpu'
        output_dir = os.path.dirname(tflite_path)
        dla_name = os.path.basename(tflite_path).replace('.tflite', '.dla')
        dla_path = os.path.join(output_dir, dla_name)
        ncc_bin = './neuronpilot-6.0.5/neuron_sdk/host/bin/ncc-tflite'
        cmd = [ncc_bin, f'--arch={device}', '--relax-fp32', tflite_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ncc-tflite failed: {result.stdout}\n{result.stderr}")
        if not os.path.exists(dla_path):
            raise RuntimeError(f"DLA 檔案未產生於 {dla_path}")
        print(f"[dla] 產生成功: {dla_path}")
        return dla_path
    except Exception as e:
        raise RuntimeError(f"TFLite to DLA conversion failed: {e}")

def onnx_to_tflite(onnx_path):
    """Convert an ONNX file to TFLite format, check shape, and test dummy inference."""
    try:
        # 使用 onnx2tf 工具將 ONNX 模型轉換為 TFLite
        subprocess.run(["onnx2tf", "-i", onnx_path, "-o", os.path.join(os.path.dirname(onnx_path), 'saved_model')], check=True, capture_output=True, text=True)
        tflite_path = os.path.join(os.path.dirname(onnx_path), 'saved_model', 'model_float32.tflite')

        # 檢查 TFLite input/output shape 是否與 ONNX 一致，允許 NCHW <-> NHWC
        onnx_model = onnx.load(onnx_path)
        onnx_input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        tflite_input_shape = interpreter.get_input_details()[0]['shape']
        if not shape_match(onnx_input_shape, tflite_input_shape):
            raise RuntimeError(f"TFLite input shape {tflite_input_shape} != ONNX input shape {onnx_input_shape} (NCHW/NHWC自動判斷)")
        print(f"[tflite] input shape: {tflite_input_shape}, output shape: {interpreter.get_output_details()[0]['shape']}")
        
        # 嘗試用隨機 dummy input 做一次推論
        dummy_input = np.random.randn(*tflite_input_shape).astype(np.float32)
        input_index = interpreter.get_input_details()[0]['index']
        interpreter.set_tensor(input_index, dummy_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        print(f"[tflite] forward success, output shape: {output_data.shape}")
        return tflite_path
    except Exception as e:
        raise RuntimeError(f"ONNX to TFLite conversion failed: {e}")