import tempfile
import importlib.util
import sys
import os

def verify_pytorch_format(pytorch_code):
    """
    驗證 PyTorch 程式碼格式與 model_entrypoint 是否可正確 import 與實例化。
    會嘗試將程式碼寫入暫存檔並 import，若失敗則丟出例外。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'user_model.py')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(pytorch_code)
        spec = importlib.util.spec_from_file_location('user_model', file_path)
        user_module = importlib.util.module_from_spec(spec)
        sys.modules['user_model'] = user_module
        try:
            spec.loader.exec_module(user_module)
        except Exception as e:
            raise RuntimeError(f"PyTorch code import failed: {e}")
    return True
