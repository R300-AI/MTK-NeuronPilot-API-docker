# 🚀 MTK NeuronPilot AI Model Porting Platform

A comprehensive web-based platform for converting AI models to MediaTek NPU-compatible DLA (Deep Learning Accelerator) form2. **Sel## 📁 Project Structure

```
MTK-NeuronPilot-API-docker/ject Structure Device Family**: Choose from Genio 510/700/1200
3. **Select Target Device**: Pick VPU, MDLA 2.0, or MDLA 3.0
4. **Download**: Click "Download DLA" to get your converted model

## 📁 Project Structureis platform streamlines the deployment of AI models to MediaTek Genio development boards with automatic conversion pipeline and compatibility checking.

## ✨ Features

### 🔄 Multi-Format Model Conversion
- **PyTorch → ONNX → TensorFlow Lite → DLA**: Complete conversion pipeline for PyTorch models
- **ONNX/TFLite Upload**: Direct upload and conversion of pre-trained models
- **Real-time Progress Tracking**: Server-sent events for live conversion progress monitoring

### 🎯 NPU Target Support
- **VPU (Vector Processing Unit)**: Optimized for vector operations
- **MDLA 2.0**: MediaTek Deep Learning Accelerator v2.0
- **MDLA 3.0**: MediaTek Deep Learning Accelerator v3.0

### 📱 Device Compatibility
| Genio Board | Supported Devices | Architecture |
|-------------|-------------------|--------------|
| Genio 510   | MDLA 3.0, VPU    | MDLA 3.0 + VPU |
| Genio 700   | MDLA 3.0, VPU    | MDLA 3.0 + VPU |
| Genio 1200  | MDLA 2.0, VPU    | MDLA 2.0 + VPU |

### 🎨 User-Friendly Interface
- **Monaco Editor**: Professional code editor with syntax highlighting
- **Interactive UI**: Tab-based interface for different input methods
- **Dynamic Dropdowns**: Automatic device selection based on compatibility
- **Real-time Logs**: Live feedback during conversion process

## 🖥️ Frontend Interface Preview

The platform provides an intuitive web interface for model conversion with real-time progress tracking and device selection capabilities.

### Interactive Model Verification Interface

![Frontend Interface](https://github.com/R300-AI/MTK-NeuronPilot-API-docker/blob/main/images/frontend.png)

**Key Features Demonstrated:**
- **Monaco Editor Integration**: Professional code editor with PyTorch syntax highlighting
- **Real-time Conversion Logs**: Live progress updates during the PyTorch → ONNX → TFLite → DLA pipeline
- **Dynamic Device Selection**: 
  - Device family dropdown (Genio 510/700/1200) with automatic compatibility detection
  - Target device selection showing available NPU options (VPU, MDLA 2.0, MDLA 3.0)
- **Conversion Status Display**: Step-by-step progress with success/failure indicators
- **Download Integration**: Direct download links for converted DLA files

The screenshot shows:
1. **Left Panel**: Monaco editor with PyTorch model code
2. **Right Panel**: Live conversion logs showing successful model verification
3. **Bottom Controls**: 
   - Genio 510 selected as the target device family
   - MDLA 3.0 dropdown opened showing available NPU targets
   - Download DLA button ready for file retrieval

This interface streamlines the entire model porting workflow from code input to DLA deployment.

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Frontend (HTML)   │    │   Backend (Flask)   │    │  Conversion Utils   │
│                     │    │                     │    │                     │
│ • Monaco Editor     │◄──►│ • Model Verification│◄──►│ • PyTorch → ONNX    │
│ • SSE Client        │    │ • File Upload       │    │ • ONNX → TFLite     │
│ • Dynamic UI        │    │ • Session Management│    │ • TFLite → DLA      │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                       │
                                       ▼
                           ┌─────────────────────┐
                           │  NeuronPilot SDK    │
                           │                     │
                           │ • ncc-tflite        │
                           │ • DLA Compilation   │
                           │ • Device Targeting  │
                           └─────────────────────┘
```

## 📋 Prerequisites

### System Requirements
- **Docker**: Version 20.0+ (recommended)
- **Python**: 3.8+ (if running without Docker)
- **Node.js**: 16+ (for development)

### Required Dependencies
- **PyTorch**: 2.0+
- **TensorFlow**: 2.8+
- **ONNX**: 1.12+
- **onnx2tf**: Latest version
- **MediaTek NeuronPilot SDK**: 6.0.5+

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/R300-AI/MTK-NeuronPilot-API-docker.git
   cd MTK-NeuronPilot-API-docker
   ```

2. **Build and run with Docker**
   ```bash
   docker build -t mtk-neuronpilot .
   docker run -p 5000:80 mtk-neuronpilot
   ```

3. **Access the platform**
   Open your browser and navigate to: `http://localhost:5000`

### Option 2: Local Development

1. **Clone and setup**
   ```bash
   git clone https://github.com/R300-AI/MTK-NeuronPilot-API-docker.git
   cd MTK-NeuronPilot-API-docker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NeuronPilot SDK**
   ```bash
   # The platform will automatically download the SDK on first run
   # Or manually place it in ./neuronpilot-6.0.5/
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## 💻 Usage Guide

### 1. PyTorch Model Conversion

1. **Select PyTorch Tab**: Choose the PyTorch option in the interface
2. **Define Your Model**: Write your PyTorch model class in the Monaco editor

   ```python
   import torch
   import torch.nn as nn

   class SimpleModel(nn.Module):
       def __init__(self):
           super(SimpleModel, self).__init__()
           self.fc1 = nn.Linear(10, 32)
           self.fc2 = nn.Linear(32, 10)
           self.relu = nn.ReLU()
       
       def forward(self, x):
           x = self.relu(self.fc1(x))
           x = self.fc2(x)
           return x
   ```

3. **Configure Parameters**:
   - **Model Entrypoint**: `SimpleModel`
   - **Input Shape**: `(1, 10)`

4. **Verify Model**: Click "Verify Model" to start the conversion pipeline

### 2. Pre-trained Model Upload

1. **Select Upload Tab**: Choose "Upload Prebuilt Model"
2. **Upload File**: Select your `.onnx` or `.tflite` file
3. **Verify Compatibility**: Click "Upload and Verify Model"

### 3. Download DLA Files

1. **Check Compatibility**: After verification, compatible devices will appear
2. **Select Device Family**: Choose from Genio 510/700/1200
3. **Select Target Device**: Pick VPU, MDLA 2.0, or MDLA 3.0
4. **Download**: Click "Download DLA" to get your converted model

## �️ Frontend Interface Preview

The platform provides an intuitive web interface for model conversion with real-time progress tracking and device selection capabilities.

### Interactive Model Verification Interface

![Frontend Interface](https://github.com/R300-AI/MTK-NeuronPilot-API-docker/blob/main/images/frontend.png)

**Key Features Demonstrated:**
- **Monaco Editor Integration**: Professional code editor with PyTorch syntax highlighting
- **Real-time Conversion Logs**: Live progress updates during the PyTorch → ONNX → TFLite → DLA pipeline
- **Dynamic Device Selection**: 
  - Device family dropdown (Genio 510/700/1200) with automatic compatibility detection
  - Target device selection showing available NPU options (VPU, MDLA 2.0, MDLA 3.0)
- **Conversion Status Display**: Step-by-step progress with success/failure indicators
- **Download Integration**: Direct download links for converted DLA files

The screenshot shows:
1. **Left Panel**: Monaco editor with PyTorch model code
2. **Right Panel**: Live conversion logs showing successful model verification
3. **Bottom Controls**: 
   - Genio 510 selected as the target device family
   - MDLA 3.0 dropdown opened showing available NPU targets
   - Download DLA button ready for file retrieval

This interface streamlines the entire model porting workflow from code input to DLA deployment.

## �📁 Project Structure

```
MTK-NeuronPilot-API-docker/
├── app.py                      # Main Flask application
├── index.html                  # Frontend interface
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── README.md                   # This documentation
├── docs/
│   └── PYTORCH_EXAMPLES.md     # PyTorch model examples
├── images/
│   ├── itri-logo.svg          # Logo assets
│   └── registry_portal.png    # Documentation images
└── utils/
    ├── file.py                # File upload and verification
    ├── format.py              # Legacy format utilities
    └── converter/
        ├── __init__.py        # Main conversion pipeline
        ├── convert.py         # Individual conversion functions
        └── format.py          # PyTorch format verification
```

## 🔧 Configuration

### Environment Variables

- `FLASK_ENV`: Set to `development` for debug mode
- `NEURONPILOT_SDK_PATH`: Custom path to NeuronPilot SDK
- `MAX_FILE_SIZE`: Maximum upload file size (default: 100MB)

### Application Settings

```python
# In app.py
USERS_ROOT_DIR = './users'          # User session directory
SESSION_EXPIRY_HOURS = 24           # Session cleanup interval
```

## 🐛 Troubleshooting

### Common Issues

#### 1. SDK Not Found
```
⚠️ DLA conversion service unavailable (SDK missing)
```
**Solution**: The platform will automatically download the NeuronPilot SDK. If manual installation is needed, place the SDK in `./neuronpilot-6.0.5/`

#### 2. Model Conversion Fails
```
❌ ONNX conversion failed: [specific error]
```
**Solutions**:
- Check PyTorch model syntax
- Verify input shape format: `(batch_size, ...dimensions)`
- Ensure model class name matches the entrypoint

#### 3. Upload File Issues
```
❌ Only .onnx or .tflite files supported
```
**Solution**: Convert your model to ONNX or TensorFlow Lite format first

#### 4. Device Compatibility
```
❌ Model cannot be ported to any DLA device
```
**Possible causes**:
- Model operations not supported by target NPU
- Model complexity exceeds device capabilities
- Input/output tensor formats incompatible

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export FLASK_ENV=development
python app.py
```

## 🔐 Security Considerations

### File Upload Security
- **File Type Validation**: Only ONNX and TFLite files accepted
- **Secure Filenames**: Uses `werkzeug.secure_filename()`
- **Session Isolation**: Each user gets isolated directory
- **Automatic Cleanup**: User directories expire after 24 hours

### Network Security
- **CORS Headers**: Configured for web browser compatibility
- **File Size Limits**: Prevents large file uploads
- **Session Management**: Unique user IDs for isolation

## 🚧 Development

### Adding New Device Support

1. **Update Device Mapping** in `index.html`:
   ```javascript
   const deviceMapping = {
     'genio510': ['mdla3.0', 'vpu'],
     'genio700': ['mdla3.0', 'vpu'],
     'genio1200': ['mdla2.0', 'vpu'],
     'new_device': ['new_npu', 'vpu']  // Add here
   };
   ```

2. **Add Conversion Function** in `utils/converter/convert.py`:
   ```python
   def tflite_to_new_npu(tflite_path):
       # Implementation for new NPU
       pass
   ```

3. **Update Pipeline** in `utils/converter/__init__.py`

### Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Commit** changes: `git commit -am 'Add feature'`
4. **Push** to branch: `git push origin feature-name`
5. **Submit** a pull request

## 📊 Performance Metrics

### Conversion Times (Approximate)
- **PyTorch → ONNX**: 5-30 seconds
- **ONNX → TFLite**: 10-60 seconds  
- **TFLite → DLA**: 15-120 seconds (depends on model complexity)

### Supported Model Types
- **Classification Models**: ✅ Full support
- **Object Detection**: ✅ Most architectures
- **Segmentation**: ✅ Common models
- **NLP Models**: ⚠️ Limited support
- **Generative Models**: ❌ Not recommended

## 📝 API Reference

### REST Endpoints

#### `POST /verify_model`
Verify and convert PyTorch model code
- **Content-Type**: `application/json`
- **Headers**: `X-User-ID: string`
- **Body**:
  ```json
  {
    "action": "verify_model",
    "pytorch_code": "string",
    "model_entrypoint": "string", 
    "input_shape": "string"
  }
  ```

#### `POST /upload_and_verify`
Upload and verify pre-trained model
- **Content-Type**: `multipart/form-data`
- **Headers**: `X-User-ID: string`
- **Body**: Form data with `upload_pretrained_file`

#### `POST /download_dla`
Download converted DLA file
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "device": "vpu|mdla2|mdla3"
  }
  ```

### Server-Sent Events

The platform uses SSE for real-time progress updates:

```javascript
// Event format
{
  "message": "Progress message",
  "error": boolean,
  "final": boolean,  // Indicates completion
  "genio510": {...}, // Device compatibility (final event only)
  "genio700": {...},
  "genio1200": {...}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- **MediaTek**: For NeuronPilot SDK and hardware support
- **ONNX Community**: For model format standards
- **TensorFlow Team**: For TensorFlow Lite framework
- **PyTorch Team**: For PyTorch framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/R300-AI/MTK-NeuronPilot-API-docker/issues)
- **Documentation**: [Wiki](https://github.com/R300-AI/MTK-NeuronPilot-API-docker/wiki)
- **Email**: support@r300.ai

---

**Made with ❤️ by the R300 AI Team**

For more examples and detailed documentation, visit our [GitHub repository](https://github.com/R300-AI/MTK-NeuronPilot-API-docker).
