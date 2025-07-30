import datetime, random
from .converter import onnx_to_tflite, tflite_to_vpu, tflite_to_mdla2, tflite_to_mdla3
import os
import json
import shutil

def verify_uploded_file(filename, save_path, user_id):
    allowed_ext = {"onnx", "tflite"}
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in allowed_ext:
        yield f'data: {{"message": "[error] Only .onnx or .tflite files are supported (received: .{ext})", "error": true}}\n\n'
        return
    yield f'data: {{"message": "[info] File uploaded: {filename}"}}\n\n'
    if ext == "onnx":
        yield f'data: {{"message": "[info] Starting ONNX to TFLite conversion..."}}\n\n'
        try:
            tflite_path = onnx_to_tflite(save_path)
            yield f'data: {{"message": "[success] Conversion completed. TFLite file generated."}}\n\n'
        except Exception as e:
            yield f'data: {{"message": "[error] Conversion failed: {str(e)}", "error": true}}\n\n'
    elif ext == "tflite":
        tflite_path = save_path
        yield f'data: {{"message": "[info] TFLite file detected. No conversion needed."}}\n\n'
    else:
        tflite_path = None
    print(tflite_path)
    # ===============================================================================
    # DLA support flags
    vpu_supported = False
    mdla2_supported = False
    mdla3_supported = False
    # Step 4: TFLite to VPU DLA
    yield f"data: {json.dumps({'message': 'Starting DLA (VPU) conversion...'})}\n\n"
    try:
        vpu_path = tflite_to_vpu(tflite_path)
        if vpu_path:
            vpu_supported = True
            yield f"data: {json.dumps({'message': 'DLA (VPU) conversion succeeded.'})}\n\n"
        else:
            yield f"data: {json.dumps({'message': 'DLA (VPU) conversion not supported.', 'error': True})}\n\n"
    except RuntimeError as e:
        yield f"data: {json.dumps({'message': f'DLA (VPU) conversion failed: {str(e)}', 'error': True})}\n\n"

    # Step 5: TFLite to MDLA2 DLA
    yield f"data: {json.dumps({'message': 'Starting DLA (MDLA2) conversion...'})}\n\n"
    try:
        mdla2_path = tflite_to_mdla2(tflite_path)
        if mdla2_path:
            mdla2_supported = True
            yield f"data: {json.dumps({'message': 'DLA (MDLA2) conversion succeeded.'})}\n\n"
        else:
            yield f"data: {json.dumps({'message': 'DLA (MDLA2) conversion not supported.', 'error': True})}\n\n"
    except RuntimeError as e:
        yield f"data: {json.dumps({'message': f'DLA (MDLA2) conversion failed: {str(e)}', 'error': True})}\n\n"

    # Step 6: TFLite to MDLA3 DLA
    yield f"data: {json.dumps({'message': 'Starting DLA (MDLA3) conversion...'})}\n\n"
    try:
        mdla3_path = tflite_to_mdla3(tflite_path)
        if mdla3_path:
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


    # 1. Reset ./users/{user_id}/export/ directory
    export_root = os.path.join('./users', str(user_id), 'export')
    if os.path.exists(export_root):
        shutil.rmtree(export_root)
    os.makedirs(export_root, exist_ok=True)

    # 2. For each successful DLA conversion, copy the file to ./users/{user_id}/export/{device}/model.dla
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
    # ===============================================================================
    

