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

import datetime
import random
import os
import json
import shutil
from .converter import onnx_to_tflite, tflite_to_vpu, tflite_to_mdla2, tflite_to_mdla3

"""
File Verification and Conversion Utilities
==========================================
檔案上傳驗證與轉換工具模組，處理 ONNX、TensorFlow Lite 等格式的模型檔案。
提供完整的檔案驗證、格式轉換與 NPU 相容性測試管線。

Functions
---------
verify_uploaded_file : 驗證上傳的模型檔案並執行 DLA 轉換管線
"""


def verify_uploaded_file(filename, save_path, user_id):
    """
    檔案上傳驗證與轉換管線
    =====================
    驗證上傳的模型檔案並執行 DLA 轉換管線，檢查與 MediaTek NPU 裝置的相容性。
    支援 ONNX 與 TensorFlow Lite 格式，自動進行格式轉換與多重 NPU 目標測試。

    Parameters
    ----------
    filename : str
        上傳檔案的原始檔名。
    save_path : str
        檔案儲存的完整路徑。
    user_id : str
        使用者會話的唯一識別碼，用於檔案管理與追蹤。

    Yields
    ------
    str
        Server-sent event 格式化的進度訊息與最終結果，包含：
        - 檔案格式驗證結果
        - ONNX → TFLite 轉換進度（如需要）
        - VPU/MDLA2/MDLA3 相容性測試結果
        - 最終轉換狀態與檔案路徑
    """
    # Validate file format
    allowed_extensions = {"onnx", "tflite"}
    file_extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    if file_extension not in allowed_extensions:
        yield f'data: {json.dumps({"message": f"❌ Only .onnx or .tflite files supported (received: .{file_extension})", "error": True, "final": True})}\n\n'
        return
    
    yield f'data: {json.dumps({"message": f"📁 File uploaded: {filename}"})}\n\n'
    
    # Initialize conversion variables
    tflite_path = None
    vpu_supported = False
    mdla2_supported = False  
    mdla3_supported = False
    vpu_path = None
    mdla2_path = None
    mdla3_path = None
    success = False
    
    # Step 1: Convert to TensorFlow Lite format if needed
    if file_extension == "onnx":
        yield f'data: {json.dumps({"message": "🔄 Starting ONNX to TFLite conversion..."})}\n\n'
        try:
            yield f'data: {json.dumps({"message": f"📂 Processing ONNX file: {save_path}"})}\n\n'
            tflite_path = onnx_to_tflite(save_path)
            yield f'data: {json.dumps({"message": f"✅ ONNX conversion completed: {tflite_path}"})}\n\n'
        except RuntimeError as e:
            yield f'data: {json.dumps({"message": f"❌ ONNX conversion failed: {str(e)}", "error": True})}\n\n'
            # Early exit on conversion failure
            yield f'data: {json.dumps({"message": "❌ Cannot proceed with DLA conversion", "error": True, "final": True})}\n\n'
            return
    elif file_extension == "tflite":
        yield f'data: {json.dumps({"message": "📝 TFLite file detected, skipping ONNX conversion"})}\n\n'
        tflite_path = save_path
    
    # Validate TFLite file path
    if not tflite_path:
        yield f'data: {json.dumps({"message": "❌ Failed to obtain TFLite file path", "error": True, "final": True})}\n\n'
        return
    
    print(f"==> TFLite file ready for DLA conversion: {tflite_path}")
    
    # Step 2: Test DLA conversions
    yield f'data: {json.dumps({"message": "🔄 Starting DLA compatibility tests..."})}\n\n'
    
    # Step 2a: TFLite to VPU DLA
    yield f'data: {json.dumps({"message": "Testing VPU compatibility..."})}\n\n'
    try:
        vpu_path = tflite_to_vpu(tflite_path)
        if vpu_path:
            vpu_supported = True
            yield f'data: {json.dumps({"message": "✅ VPU conversion succeeded"})}\n\n'
        else:
            yield f'data: {json.dumps({"message": "❌ VPU conversion not supported", "error": True})}\n\n'
    except RuntimeError as e:
        yield f'data: {json.dumps({"message": f"❌ VPU conversion failed: {str(e)}", "error": True})}\n\n'
    
    # Step 2b: TFLite to MDLA2 DLA  
    yield f'data: {json.dumps({"message": "Testing MDLA 2.0 compatibility..."})}\n\n'
    try:
        mdla2_path = tflite_to_mdla2(tflite_path)
        if mdla2_path:
            mdla2_supported = True
            yield f'data: {json.dumps({"message": "✅ MDLA 2.0 conversion succeeded"})}\n\n'
        else:
            yield f'data: {json.dumps({"message": "❌ MDLA 2.0 conversion not supported", "error": True})}\n\n'
    except RuntimeError as e:
        yield f'data: {json.dumps({"message": f"❌ MDLA 2.0 conversion failed: {str(e)}", "error": True})}\n\n'

    # Step 2c: TFLite to MDLA3 DLA
    yield f'data: {json.dumps({"message": "Testing MDLA 3.0 compatibility..."})}\n\n'
    try:
        mdla3_path = tflite_to_mdla3(tflite_path)
        if mdla3_path:
            mdla3_supported = True
            yield f'data: {json.dumps({"message": "✅ MDLA 3.0 conversion succeeded"})}\n\n'
        else:
            yield f'data: {json.dumps({"message": "❌ MDLA 3.0 conversion not supported", "error": True})}\n\n'
    except RuntimeError as e:
        yield f'data: {json.dumps({"message": f"❌ MDLA 3.0 conversion failed: {str(e)}", "error": True})}\n\n'

    # Step 3: Generate compatibility summary
    yield f'data: {json.dumps({"message": "📊 Generating compatibility summary..."})}\n\n'
    
    # Check if NeuronPilot SDK is available for status display
    ncc_binary = './neuronpilot-6.0.5/neuron_sdk/host/bin/ncc-tflite'
    sdk_available = os.path.exists(ncc_binary)
    
    # Generate status messages based on SDK availability
    if sdk_available:
        vpu_status = '✅ Supported' if vpu_supported else '❌ Not Supported'
        mdla2_status = '✅ Supported' if mdla2_supported else '❌ Not Supported'
        mdla3_status = '✅ Supported' if mdla3_supported else '❌ Not Supported'
    else:
        vpu_status = '✅ Supported' if vpu_supported else '⚠️ SDK Missing'
        mdla2_status = '✅ Supported' if mdla2_supported else '⚠️ SDK Missing'
        mdla3_status = '✅ Supported' if mdla3_supported else '⚠️ SDK Missing'

    # Display compatibility results
    yield f'data: {json.dumps({"message": "============ DLA Compatibility ============"})}\n\n'
    yield f'data: {json.dumps({"message": f"VPU:      {vpu_status}"})}\n\n'
    yield f'data: {json.dumps({"message": f"MDLA 2.0: {mdla2_status}"})}\n\n'
    yield f'data: {json.dumps({"message": f"MDLA 3.0: {mdla3_status}"})}\n\n'
    yield f'data: {json.dumps({"message": "==========================================="})}\n\n'

    # Step 4: Generate final conclusion
    success = False
    if not any([vpu_supported, mdla2_supported, mdla3_supported]):
        if not sdk_available:
            yield f'data: {json.dumps({"message": "⚠️ DLA conversion service unavailable (SDK missing)", "error": True})}\n\n'
        else:
            yield f'data: {json.dumps({"message": "❌ Model cannot be ported to any DLA device", "error": True})}\n\n'
    else:
        # List supported devices
        supported_devices = []
        if vpu_supported: supported_devices.append('VPU')
        if mdla2_supported: supported_devices.append('MDLA 2.0')
        if mdla3_supported: supported_devices.append('MDLA 3.0')
        
        if supported_devices:
            supported_str = ', '.join(supported_devices)
            yield f'data: {json.dumps({"message": f"✅ Model compatible with: {supported_str}"})}\n\n'
            success = True

    # Step 5: Send final response for frontend dropdown updates
    final_response = {
        'final': True,
        'success': success,
        'vpu_supported': vpu_supported,
        'mdla2_supported': mdla2_supported,
        'mdla3_supported': mdla3_supported,
        'genio510': {'vpu': False, 'mdla2': False, 'mdla3': False},
        'genio700': {'vpu': False, 'mdla2': False, 'mdla3': False},
        'genio1200': {'vpu': False, 'mdla2': False, 'mdla3': False},
    }

    # Map device support to Genio board compatibility
    if vpu_supported:
        final_response['genio510']['vpu'] = True
        final_response['genio700']['vpu'] = True  
        final_response['genio1200']['vpu'] = True
    if mdla2_supported:
        final_response['genio1200']['mdla2'] = True
    if mdla3_supported:
        final_response['genio510']['mdla3'] = True
        final_response['genio700']['mdla3'] = True
    
    print(f"==> Final response: {final_response}")
    yield f'data: {json.dumps(final_response)}\n\n'