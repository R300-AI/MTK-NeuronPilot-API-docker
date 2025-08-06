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

import tempfile
import importlib.util
import sys
import os

"""
PyTorch Code Format Verification Utilities
==========================================
PyTorch 程式碼格式驗證工具，提供基礎的語法檢查與模組匯入測試。
確保使用者程式碼無語法錯誤且可正常載入。

Functions
---------
verify_pytorch_format : PyTorch 程式碼格式驗證
"""

def verify_pytorch_format(pytorch_code):
    """
    PyTorch 程式碼格式驗證
    ====================
    驗證 PyTorch 程式碼語法正確性與模組可匯入性。
    建立暫存檔案執行 import 測試，確保程式碼無語法錯誤且可正常執行。

    Parameters
    ----------
    pytorch_code : str
        待驗證的 PyTorch 程式碼字串，應包含完整的模型定義。

    Returns
    -------
    bool
        驗證成功時返回 True。

    Raises
    ------
    RuntimeError
        當程式碼語法錯誤或 import 失敗時拋出，包含詳細的錯誤訊息。
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
