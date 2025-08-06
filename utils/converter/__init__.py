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

import json
import os
import shutil
from .format import verify_pytorch_format
from .convert import onnx_to_tflite, tflite_to_vpu, tflite_to_mdla2, tflite_to_mdla3

"""
PyTorch Model Conversion Pipeline
=================================
PyTorch 模型轉換管線核心模組，提供完整的 PyTorch → ONNX → TensorFlow Lite → DLA 轉換流程。
支援即時進度追蹤、錯誤處理與多重 NPU 目標相容性測試。

Functions
---------
convert_pytorch_to_tflite : 執行完整的 PyTorch 模型轉換管線
"""


def convert_pytorch_to_tflite(user_id, pytorch_code, model_entrypoint, input_shape):
    """
    PyTorch Model Conversion Pipeline
    =================================
    將 PyTorch 模型程式碼轉換為 MediaTek NPU 相容的 DLA 格式，支援即時進度更新。
    執行完整的轉換流程：PyTorch → ONNX → TensorFlow Lite → DLA（VPU/MDLA2/MDLA3）。

    Parameters
    ----------
    user_id : str
        使用者會話的唯一識別碼，用於檔案管理與追蹤。
    pytorch_code : str
        PyTorch 模型類別定義程式碼。
    model_entrypoint : str
        要實例化的模型類別名稱。
    input_shape : str
        輸入張量形狀字串，例如 "(1, 10)" 或 "(1, 3, 224, 224)"。

    Yields
    ------
    str
        Server-sent event 格式化的進度訊息與最終結果，包含轉換狀態、錯誤訊息和相容性測試結果。
    """
    success = True
    
    # Step 1: Initialize conversion process
    try:
        yield f'data: {json.dumps({"message": "🚀 PyTorch conversion pipeline started"})}\n\n'
        yield f'data: {json.dumps({"message": f"📝 Model class: {model_entrypoint}"})}\n\n'
        yield f'data: {json.dumps({"message": f"📐 Input shape: {input_shape}"})}\n\n'
    except Exception as e:
        success = False
        yield f'data: {json.dumps({"message": f"❌ Initialization failed: {str(e)}", "error": True})}\n\n'

    # Step 2: PyTorch to ONNX conversion
    yield f'data: {json.dumps({"message": "🔄 Starting PyTorch → ONNX conversion..."})}\n\n'
    try:
        onnx_path = verify_pytorch_format(user_id, pytorch_code, model_entrypoint, input_shape)
        yield f'data: {json.dumps({"message": "✅ ONNX conversion completed"})}\n\n'
    except RuntimeError as e:
        success = False
        yield f'data: {json.dumps({"message": f"❌ ONNX conversion failed: {str(e)}", "error": True})}\n\n'
        return

    # Step 3: ONNX to TensorFlow Lite conversion
    yield f'data: {json.dumps({"message": "🔄 Starting ONNX → TensorFlow Lite conversion..."})}\n\n'
    try:
        tflite_path = onnx_to_tflite(onnx_path)
        yield f'data: {json.dumps({"message": "✅ TensorFlow Lite conversion completed"})}\n\n'
    except RuntimeError as e:
        success = False
        yield f'data: {json.dumps({"message": f"❌ TensorFlow Lite conversion failed: {str(e)}", "error": True})}\n\n'
        return

    # Step 4: DLA format conversions
    yield f'data: {json.dumps({"message": "🔄 Testing NPU device compatibility..."})}\n\n'
    
    # Initialize DLA support tracking
    vpu_supported = False
    mdla2_supported = False
    mdla3_supported = False
    vpu_path = None
    mdla2_path = None
    mdla3_path = None

    # Step 4a: TFLite to VPU DLA
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

    
    # Step 4b: TFLite to MDLA2 DLA
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

    # Step 4c: TFLite to MDLA3 DLA
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

    # Step 5: Generate compatibility summary
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

    # Step 6: Process and organize DLA files (if any were generated)
    try:
        # Create organized export directory structure
        export_root = os.path.join('./users', str(user_id), 'export')
        if os.path.exists(export_root):
            shutil.rmtree(export_root)
        os.makedirs(export_root, exist_ok=True)

        # Copy successful conversions to organized structure
        if vpu_supported and vpu_path:
            vpu_export_dir = os.path.join(export_root, 'vpu')
            os.makedirs(vpu_export_dir, exist_ok=True)
            vpu_export_path = os.path.join(vpu_export_dir, 'model.dla')
            shutil.copyfile(vpu_path, vpu_export_path)
            
        if mdla2_supported and mdla2_path:
            mdla2_export_dir = os.path.join(export_root, 'mdla2')
            os.makedirs(mdla2_export_dir, exist_ok=True)
            mdla2_export_path = os.path.join(mdla2_export_dir, 'model.dla')
            shutil.copyfile(mdla2_path, mdla2_export_path)
            
        if mdla3_supported and mdla3_path:
            mdla3_export_dir = os.path.join(export_root, 'mdla3')
            os.makedirs(mdla3_export_dir, exist_ok=True)
            mdla3_export_path = os.path.join(mdla3_export_dir, 'model.dla')
            shutil.copyfile(mdla3_path, mdla3_export_path)

    except Exception as e:
        yield f'data: {json.dumps({"message": f"⚠️ File organization warning: {str(e)}", "error": True})}\n\n'

    # Step 7: Generate final conclusion
    success = False
    supported_devices = []
    if vpu_supported: supported_devices.append('VPU')
    if mdla2_supported: supported_devices.append('MDLA 2.0')
    if mdla3_supported: supported_devices.append('MDLA 3.0')
    
    if supported_devices:
        supported_str = ', '.join(supported_devices)
        yield f'data: {json.dumps({"message": f"✅ Model compatible with: {supported_str}"})}\n\n'
        success = True
    else:
        if not sdk_available:
            yield f'data: {json.dumps({"message": "⚠️ DLA conversion service unavailable (SDK missing)", "error": True})}\n\n'
        else:
            yield f'data: {json.dumps({"message": "❌ Model cannot be ported to any DLA device", "error": True})}\n\n'

    # Step 8: Send final response for frontend dropdown updates
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