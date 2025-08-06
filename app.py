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
import json
import time
import shutil
from flask import Flask, request, send_from_directory, jsonify, Response
from werkzeug.utils import secure_filename
import torch
import tensorflow as tf

from utils.file import verify_uploaded_file
from utils.converter import convert_pytorch_to_tflite

"""
MTK NeuronPilot AI Model Porting Platform
=========================================
MediaTek NPU 相容性測試與模型轉換平台，支援 PyTorch 模型自動轉換為 DLA 格式。
提供 Web 介面進行模型上傳、驗證與下載，支援 VPU、MDLA 2.0、MDLA 3.0 等 NPU 裝置。

Features
--------
- PyTorch → ONNX → TensorFlow Lite → DLA 轉換管線
- 即時轉換進度追蹤與錯誤回報
- 多種 NPU 目標裝置相容性測試
- 自動檔案清理與會話管理
- Monaco Editor 程式碼編輯介面

Supported Devices
-----------------
- Genio 510 (VPU)
- Genio 700 (MDLA 2.0) 
- Genio 1200 (MDLA 3.0)
"""

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Configuration constants
USERS_ROOT_DIR = './users'
SESSION_EXPIRY_HOURS = 24

def cleanup_expired_users():
    """
    過期使用者目錄清理
    ================
    清理超過 24 小時未修改的使用者目錄，避免磁碟空間浪費。
    自動移除舊的轉換檔案與暫存資料，維持系統效能。

    Process
    -------
    1. 檢查 USERS_ROOT_DIR 目錄是否存在
    2. 遍歷所有使用者子目錄
    3. 比較目錄修改時間與當前時間
    4. 移除超過 SESSION_EXPIRY_HOURS 的目錄
    5. 記錄清理操作與例外處理

    Note
    ----
    每次主頁面請求時自動執行，確保定期清理過期檔案。
    """
    if not os.path.exists(USERS_ROOT_DIR):
        return
    
    current_time = time.time()
    expiry_seconds = SESSION_EXPIRY_HOURS * 60 * 60
    
    for user_dir in os.listdir(USERS_ROOT_DIR):
        user_path = os.path.join(USERS_ROOT_DIR, user_dir)
        if not os.path.isdir(user_path):
            continue
            
        try:
            modified_time = os.path.getmtime(user_path)
            if current_time - modified_time > expiry_seconds:
                shutil.rmtree(user_path)
                print(f"[CLEANUP] Removed expired user directory: {user_path}")
        except Exception as e:
            print(f"[CLEANUP] Error processing {user_path}: {e}")


def inject_version_info(html_content):
    """
    HTML 模板版本資訊注入
    ====================
    將 PyTorch 與 TensorFlow 版本資訊注入到 HTML 模板中的分頁按鈕。
    動態更新分頁標題，讓使用者了解當前環境的套件版本。

    Parameters
    ----------
    html_content : str
        原始的 HTML 模板內容。

    Returns
    -------
    str
        注入版本資訊後的 HTML 內容，分頁按鈕將顯示對應的套件版本。
    """
    pytorch_version = torch.__version__
    tensorflow_version = tf.__version__
    
    # Update PyTorch tab button with version info
    html_content = html_content.replace(
        '<button type="button" class="tab-btn active" id="tab-pytorch" onclick="switchTab(\'pytorch\')">PyTorch</button>',
        f'<button type="button" class="tab-btn active" id="tab-pytorch" onclick="switchTab(\'pytorch\')">PyTorch={pytorch_version}</button>'
    )
    
    # Update TensorFlow tab button with version info
    html_content = html_content.replace(
        '<button type="button" class="tab-btn" id="tab-tf" onclick="switchTab(\'tf\')">TensorFlow</button>',
        f'<button type="button" class="tab-btn" id="tab-tf" onclick="switchTab(\'tf\')">TensorFlow={tensorflow_version}</button>'
    )
    
    return html_content


@app.route('/', methods=['GET', 'POST'])
def api_index():
    """
    主要路由處理器
    ============
    處理 Web 介面的 GET 與 POST 請求，支援多種檔案格式與轉換模式。
    GET 請求提供主介面，POST 請求根據內容類型路由到對應的處理器。

    HTTP Methods
    ------------
    GET : 返回主要的 HTML 介面，包含版本資訊與清理過期使用者目錄
    POST : 根據請求類型路由處理：
           - JSON 請求：PyTorch 模型驗證與轉換
           - 檔案上傳：ONNX/TFLite 檔案處理

    Returns
    -------
    GET : 注入版本資訊的 HTML 模板
    POST : 重導向到對應的處理函數 (api_convert_pytorch 或 api_upload_file)
    """
    # Perform cleanup of expired user directories
    cleanup_expired_users()
    
    # Handle POST requests
    if request.method == 'POST':
        print(f'==> Request method: {request.method}')
        
        if request.is_json:
            # Handle JSON requests (PyTorch model verification)
            print('==> Processing JSON request')
            data = request.get_json()
            action = data.get('action', 'convert_tflite')
            
            if action == 'verify_model':
                print('==> Routing to PyTorch model verification')
                return api_verify_model()
                
        else:
            # Handle form data requests (file upload and conversion)
            print('==> Processing form data request')
            action = request.form.get('action', '')
            print(f'==> Action: {action}')
            
            if action == 'upload_and_verify':
                print('==> Routing to upload and verify')
                return upload_and_verify()
            elif action == 'convert_tflite':
                print('==> Routing to DLA download')
                return download_dla()
    
    # Serve the main HTML interface
    try:
        with open('index.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
            html_content = inject_version_info(html_content)
            return html_content
    except FileNotFoundError:
        return jsonify({"error": "Interface template not found"}), 500

@app.route('/upload_and_verify', methods=['POST'])
def upload_and_verify():
    """
    檔案上傳與驗證處理器
    ==================
    處理預訓練模型檔案上傳，並執行 NPU 相容性驗證管線。
    支援 ONNX、TensorFlow Lite 等格式，提供即時轉換進度與結果回饋。

    Request Format
    --------------
    POST multipart/form-data
    - upload_pretrained_file : 上傳的模型檔案
    - X-User-ID header : 使用者會話識別碼

    Returns
    -------
    Response
        Server-sent events 串流，包含轉換進度、錯誤訊息和最終結果。
        Content-Type: text/event-stream

    Response Format
    ---------------
    data: {"message": "進度訊息", "error": bool, "final": bool}
    """
    user_id = request.headers.get('X-User-ID')
    
    # Validate file upload
    if 'upload_pretrained_file' not in request.files:
        def error_response():
            yield f'data: {json.dumps({"message": "❌ No file received (upload_pretrained_file)", "error": True, "final": True})}\n\n'
        return Response(
            error_response(), 
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
    
    # Process uploaded file
    file = request.files['upload_pretrained_file']
    filename = secure_filename(file.filename)
    
    # Create user directory and save file
    save_dir = f'./users/{user_id}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    file.save(save_path)
    
    # Start verification process
    return Response(
        verify_uploaded_file(filename, save_path, user_id),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/verify_model', methods=['POST'])
def api_verify_model():
    """
    PyTorch 模型驗證與轉換處理器
    ==========================
    接收 PyTorch 模型程式碼並執行完整的轉換管線。
    處理流程：PyTorch → ONNX → TensorFlow Lite → DLA 格式。

    Request Format
    --------------
    POST application/json
    - pytorch_code : PyTorch 模型類別定義程式碼
    - model_entrypoint : 模型類別名稱 (預設: "SimpleModel")
    - input_shape : 輸入張量形狀 (預設: "(1, 10)")
    - tf_code : TensorFlow 程式碼 (預留功能)
    - X-User-ID header : 使用者會話識別碼

    Returns
    -------
    Response
        Server-sent events 串流，包含轉換進度、錯誤訊息和最終結果。
        Content-Type: text/event-stream
    """
    data = request.get_json()
    user_id = request.headers.get('X-User-ID')
    
    # Extract request parameters
    pytorch_code = data.get('pytorch_code', '')
    tf_code = data.get('tf_code', '')  # Reserved for future TensorFlow support
    model_entrypoint = data.get('model_entrypoint', 'SimpleModel')
    input_shape = data.get('input_shape', '(1, 10)')

    # Start conversion process
    return Response(
        convert_pytorch_to_tflite(
            user_id=user_id,
            pytorch_code=pytorch_code,
            model_entrypoint=model_entrypoint,
            input_shape=input_shape
        ),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/download_dla', methods=['POST'])
def download_dla():
    """
    DLA 檔案下載處理器
    ================
    搜尋使用者目錄中的 DLA 檔案並提供下載服務。
    支援 VPU、MDLA 2.0、MDLA 3.0 格式的 DLA 檔案下載。

    Request Format
    --------------
    POST application/json
    - device : 目標裝置類型 ("vpu", "mdla2", "mdla3")
    - X-User-ID header : 使用者會話識別碼

    Returns
    -------
    Response
        成功時返回 DLA 檔案下載，失敗時返回 JSON 錯誤訊息。
        Content-Type: application/octet-stream (下載) 或 application/json (錯誤)

    Error Codes
    -----------
    400 : 無效的裝置類型
    404 : DLA 檔案不存在
    """
    print("==> DLA download API called")
    
    user_id = request.headers.get('X-User-ID')
    data = request.get_json()
    target_device = data.get('device')  # vpu, mdla2, mdla3
    
    # Device type to file suffix mapping
    device_suffix_map = {
        'vpu': '_vpu.dla',
        'mdla2': '_mdla2.dla', 
        'mdla3': '_mdla3.dla'
    }
    
    # Validate device type
    if target_device not in device_suffix_map:
        print(f"==> Invalid device type: {target_device}")
        return jsonify({"error": "Invalid device type"}), 400
    
    # Search for DLA file in user directory (recursive search)
    user_dir = f'./users/{user_id}'
    dla_suffix = device_suffix_map[target_device]
    dla_file = None
    
    if os.path.exists(user_dir):
        for root, dirs, files in os.walk(user_dir):
            for filename in files:
                if filename.endswith(dla_suffix):
                    dla_file = os.path.join(root, filename)
                    break
            if dla_file:  # Exit outer loop if file found
                break
    
    # Validate file existence
    if not dla_file or not os.path.exists(dla_file):
        print(f"==> DLA file not found for device {target_device} in {user_dir}")
        return jsonify({"error": "Requested DLA file not found"}), 404

    # Serve the file
    print(f"==> Serving DLA file: {dla_file}")
    return send_from_directory(
        directory=os.path.dirname(dla_file),
        path=os.path.basename(dla_file),
        as_attachment=True,
        download_name=f'model_{target_device}.dla',
        mimetype='application/octet-stream'
    )


if __name__ == '__main__':
    """
    應用程式進入點
    ============
    啟動 Flask 開發伺服器，監聽所有網路介面的 8091 埠。
    生產環境建議使用 Gunicorn 等正式的 WSGI 伺服器。

    Server Configuration
    -------------------
    - Host: 0.0.0.0 (所有網路介面)
    - Debug: False (關閉除錯模式)
    """
    app.run(host='0.0.0.0', port=8092, debug=False)
