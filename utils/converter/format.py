import os
import sys
import subprocess

def verify_pytorch_format(user_id, pytorch_code, model_entrypoint, input_shape):
    if not pytorch_code.strip():
        raise RuntimeError('❌ PyTorch 程式碼為空')
    try:
        # 1. 驗證語法與 import
        result = subprocess.run([
            sys.executable, '-c', pytorch_code
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"PyTorch code import failed: {result.stderr}\n{result.stdout}")

        # 2. 自動補上模型建立、dummy input、ONNX匯出，並存成.py
        user_dir = os.path.join('.', 'users', str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        onnx_path = os.path.join(user_dir, 'model.onnx')
        full_code = pytorch_code.rstrip() + f"\nmodel = {model_entrypoint}()\nmodel.eval()\ndummy_input = torch.randn{input_shape}\ntorch.onnx.export(model, dummy_input, r'{onnx_path}', opset_version=11)\n"
        export_path = os.path.join(user_dir, 'export.py')
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(full_code)
        print("\n===== [export.py generated for user_id: {}] =====".format(user_id))
        print(full_code)
        print("===== [end of export.py] =====\n")
        result2 = subprocess.run([sys.executable, export_path], capture_output=True, text=True)
        if result2.returncode != 0:
            raise RuntimeError(f"export.py failed: {result2.stderr}\n{result2.stdout}")
        # 3. 先比對 onnx 檔案的 input/output shape，再用 onnxruntime forward
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np
            shape = eval(input_shape) if isinstance(input_shape, str) else input_shape
            onnx_model = onnx.load(onnx_path)
            input_tensors = onnx_model.graph.input
            output_tensors = onnx_model.graph.output
            # 只檢查第一個 input/output
            onnx_input_shape = [d.dim_value for d in input_tensors[0].type.tensor_type.shape.dim]
            if tuple(onnx_input_shape) != tuple(shape):
                raise RuntimeError(f"ONNX input shape {onnx_input_shape} != 指定 shape {shape}")
            print(f"[onnx] input shape: {onnx_input_shape}, output shape: {[d.dim_value for d in output_tensors[0].type.tensor_type.shape.dim]}")
            dummy_input = np.random.randn(*shape).astype(np.float32)
            sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            input_name = sess.get_inputs()[0].name
            output = sess.run(None, {input_name: dummy_input})
            print(f"[onnxruntime] forward success, output shape: {[o.shape for o in output]}")
        except Exception as e:
            raise RuntimeError(f"ONNX 檔案 I/O 檢查或推論測試失敗: {e}")
        return onnx_path
    except Exception as e:
        raise RuntimeError(f"PyTorch code import or export failed: {e}")