from .format import verify_pytorch_format
from .convert import onnx_to_tflite, tflite_to_vpu, tflite_to_mdla2, tflite_to_mdla3
import json, subprocess, os

def convert_pytorch_to_tflite(user_id, pytorch_code, model_entrypoint, input_shape):
    success = True
    # Step 1: Receive request
    try:
        yield f"data: {json.dumps({'message': 'PyTorch conversion request received.'})}\n\n"
        yield f"data: {json.dumps({'message': f'Model entrypoint: {model_entrypoint}'})}\n\n"
        yield f"data: {json.dumps({'message': f'Input shape: {input_shape}'})}\n\n"
    except Exception as e:
        err_msg = f"Exception: {str(e)}\n"
        success = False
        yield f"data: {json.dumps({'message': err_msg, 'error': True})}\n\n"

    # Step 2: PyTorch to ONNX
    yield f"data: {json.dumps({'message': 'Starting ONNX conversion...'})}\n\n"
    try:
        onnx_path = verify_pytorch_format(user_id, pytorch_code, model_entrypoint, input_shape)
        yield f"data: {json.dumps({'message': 'ONNX conversion completed.'})}\n\n"
    except RuntimeError as e:
        success = False
        yield f"data: {json.dumps({'message': f'ONNX conversion failed: {str(e)}', 'error': True})}\n\n"

    # Step 3: ONNX to TFLite
    yield f"data: {json.dumps({'message': 'Starting TFLite conversion...'})}\n\n"
    try:
        tflite_path = onnx_to_tflite(onnx_path)
        yield f"data: {json.dumps({'message': 'TFLite conversion completed.'})}\n\n"
    except RuntimeError as e:
        success = False
        yield f"data: {json.dumps({'message': f'TFLite conversion failed: {str(e)}', 'error': True})}\n\n"


    # DLA support flags
    vpu_supported = False
    mdla2_supported = False
    mdla3_supported = False

    # Step 4: TFLite to VPU DLA
    yield f"data: {json.dumps({'message': 'Starting DLA (VPU) conversion...'})}\n\n"
    try:
        dla_path = tflite_to_vpu(tflite_path)
        if dla_path:
            vpu_supported = True
            yield f"data: {json.dumps({'message': 'DLA (VPU) conversion succeeded.'})}\n\n"
        else:
            yield f"data: {json.dumps({'message': 'DLA (VPU) conversion not supported.', 'error': True})}\n\n"
    except RuntimeError as e:
        yield f"data: {json.dumps({'message': f'DLA (VPU) conversion failed: {str(e)}', 'error': True})}\n\n"

    # Step 5: TFLite to MDLA2 DLA
    yield f"data: {json.dumps({'message': 'Starting DLA (MDLA2) conversion...'})}\n\n"
    try:
        dla_path = tflite_to_mdla2(tflite_path)
        if dla_path:
            mdla2_supported = True
            yield f"data: {json.dumps({'message': 'DLA (MDLA2) conversion succeeded.'})}\n\n"
        else:
            yield f"data: {json.dumps({'message': 'DLA (MDLA2) conversion not supported.', 'error': True})}\n\n"
    except RuntimeError as e:
        yield f"data: {json.dumps({'message': f'DLA (MDLA2) conversion failed: {str(e)}', 'error': True})}\n\n"

    # Step 6: TFLite to MDLA3 DLA
    yield f"data: {json.dumps({'message': 'Starting DLA (MDLA3) conversion...'})}\n\n"
    try:
        dla_path = tflite_to_mdla3(tflite_path)
        if dla_path:
            mdla3_supported = True
            yield f"data: {json.dumps({'message': 'DLA (MDLA3) conversion succeeded.'})}\n\n"
        else:
            yield f"data: {json.dumps({'message': 'DLA (MDLA3) conversion not supported.', 'error': True})}\n\n"
    except RuntimeError as e:
        yield f"data: {json.dumps({'message': f'DLA (MDLA3) conversion failed: {str(e)}', 'error': True})}\n\n"


    # DLA summary, yield line by line (fix emoji in f-string)
    vpu_status = '\u2705 Supported' if vpu_supported else '\u274C Not Supported'
    mdla2_status = '\u2705 Supported' if mdla2_supported else '\u274C Not Supported'
    mdla3_status = '\u2705 Supported' if mdla3_supported else '\u274C Not Supported'

    yield f"data: {json.dumps({'message': '============ Portable DLA  ============'})}\n\n"
    yield f"data: {json.dumps({'message': f'VPU:    {vpu_status}'})}\n\n"
    yield f"data: {json.dumps({'message': f'MDLA2:  {mdla2_status}'})}\n\n"
    yield f"data: {json.dumps({'message': f'MDLA3:  {mdla3_status}'})}\n\n"
    yield f"data: {json.dumps({'message': '======================================='})}\n\n"

    # DLA conclusion
    supported = []
    if vpu_supported: supported.append('VPU')
    if mdla2_supported: supported.append('MDLA2')
    if mdla3_supported: supported.append('MDLA3')
    if supported:
        msg = "\u2705 Model can be ported to: " + ", ".join(supported) + "."
        yield f"data: {json.dumps({'message': msg})}\n\n"
        success = True
    else:
        msg = "\u274C Model cannot be ported to any DLA device."
        yield f"data: {json.dumps({'message': msg, 'error': True})}\n\n"
        success = False

    yield f"data: {json.dumps({'final': True, 'success': success})}\n\n"
    return

