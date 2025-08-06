# -*- coding: utf-8 -*-
"""
版權所有 © 2025 工業技術研究院 (ITRI) 及貢獻者。
保留所有權利。

本檔案由 Microsoft 訂閱的 GitHub Copilot AI 助理協助產生與優化，部分內容經人工審閱與修正。

本程式碼僅供學術研究與內部使用，未經授權不得用於商業用途。

重新發佈與使用（無論原始或二進位形式，是否經過修改）僅限於下列條件下：

* 原始碼之再發佈必須保留上述版權聲明、條件列表及下列免責聲明。
* 二進位形式之再發佈必須於相關文件或其他資料中重現上述版權聲明、條件列表及下列免責聲明。
* 未經事先書面同意，不得使用工業技術研究院 (ITRI) 或貢獻者之名稱為本軟體衍生產品背書或推廣。

本軟體以「現狀」提供，不附任何明示或暗示之保證，包括但不限於適售性及特定用途之適用性。工業技術研究院 (ITRI) 或貢獻者對於因本軟體使用或無法使用所生之任何直接、間接、附帶、特殊、懲罰性或衍生性損害（包括但不限於替代商品或服務之取得、使用損失、資料遺失、營業中斷等），無論於任何理論下（契約、侵權或其他），即使已被告知可能發生該等損害，亦不負任何責任。
"""

import os
import sys
import subprocess

"""
PyTorch Model Format Verification
=================================
PyTorch 模型格式驗證模組，提供程式碼語法檢查、模型建立測試與 ONNX 匯出功能。
確保使用者提供的 PyTorch 程式碼可正確執行並成功轉換為 ONNX 格式。

Functions
---------
verify_pytorch_format : PyTorch 模型格式驗證與 ONNX 匯出
"""

def verify_pytorch_format(user_id, pytorch_code, model_entrypoint, input_shape):
    """
    PyTorch 模型格式驗證與 ONNX 匯出
    ==============================
    驗證 PyTorch 程式碼語法、模型類別可實例化性，並自動匯出為 ONNX 格式。
    執行完整的驗證流程：語法檢查 → 模型建立 → ONNX 匯出 → 形狀驗證 → 推論測試。

    Parameters
    ----------
    user_id : str
        使用者會話的唯一識別碼，用於建立專屬工作目錄。
    pytorch_code : str
        PyTorch 模型類別定義程式碼，必須包含完整的 import 與類別定義。
    model_entrypoint : str
        要實例化的模型類別名稱，該類別必須在 pytorch_code 中定義。
    input_shape : str or tuple
        輸入張量形狀，字串格式如 "(1, 3, 224, 224)" 或直接傳入 tuple。

    Returns
    -------
    str
        成功匯出的 ONNX 檔案完整路徑。

    Raises
    ------
    RuntimeError
        當程式碼為空、語法錯誤、模型建立失敗、ONNX 匯出失敗或形狀不符時拋出。
    """
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