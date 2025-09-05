# 🚀 MTK NeuronPilot AI 模型移植平台

基於網頁的AI模型轉換平台，可將AI模型轉換為MediaTek NPU相容的DLA (Deep Learning Accelerator) 格式。

## ✨ 功能特色

- **PyTorch → ONNX → TensorFlow Lite → DLA**：PyTorch模型完整轉換管線
- **ONNX/TFLite上傳**：直接上傳並轉換預訓練模型
- **即時進度追蹤**：實時監控轉換進度
- **多NPU支援**：支援VPU、MDLA 2.0、MDLA 3.0

## 📱 支援裝置

| Genio開發板 | 支援的NPU |
|-------------|-----------|
| Genio 510   | MDLA 3.0, VPU |
| Genio 700   | MDLA 3.0, VPU |
| Genio 1200  | MDLA 2.0, VPU |

## 🚀 快速開始

### 系統需求
- **Docker**：版本 20.0+ 
- **Git**：用於複製專案

### 安裝步驟

1. **複製專案**
   ```bash
   git clone https://github.com/R300-AI/MTK-NeuronPilot-API-docker.git
   cd MTK-NeuronPilot-API-docker
   ```

2. **下載 NeuronPilot SDK**
   ```bash
   wget https://itriaihub.blob.core.windows.net/github-download-resources/repository/ITRI-AI-Hub/neuronpilot-6.0.5_x86_64.tar.gz
   ```

3. **建置並執行 Docker 容器**
   ```bash
   docker build -t mtk-neuronpilot .
   docker run -p 5000:80 mtk-neuronpilot
   ```

4. **訪問平台**
   開啟瀏覽器並前往：`http://localhost:5000`

## 💻 使用方法

### PyTorch 模型轉換

1. **選擇 PyTorch 分頁**
2. **撰寫模型程式碼**：
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

3. **設定參數**：
   - **模型進入點**：`SimpleModel`
   - **輸入形狀**：`(1, 10)`

4. **點擊「驗證模型」**開始轉換

### 預訓練模型上傳

1. **選擇「上傳預建模型」分頁**
2. **上傳檔案**：選擇您的 `.onnx` 或 `.tflite` 檔案
3. **點擊「上傳並驗證模型」**

### 下載轉換後的模型

1. 轉換成功後，從下拉選單選擇您的 **Genio開發板**
2. 選擇 **目標NPU**（VPU、MDLA 2.0 或 MDLA 3.0）
3. **點擊「下載DLA」**取得轉換後的模型

## 🔧 介面預覽

![前端介面](https://github.com/R300-AI/MTK-NeuronPilot-API-docker/blob/main/images/frontend.png)

網頁介面提供：
- **程式碼編輯器**：撰寫PyTorch模型定義
- **檔案上傳**：上傳ONNX/TFLite模型
- **即時日誌**：監控轉換進度
- **裝置選擇**：選擇目標Genio開發板和NPU
- **下載功能**：取得轉換後的DLA檔案

## 🐛 故障排除

### 常見問題

**SDK下載問題：**
```bash
# 如果 wget 失敗，嘗試使用 curl：
curl -L -O https://itriaihub.blob.core.windows.net/github-download-resources/repository/ITRI-AI-Hub/neuronpilot-6.0.5_x86_64.tar.gz
```

**Docker建置失敗：**
- 確保Docker正在執行
- 檢查網路連線以下載相依套件
- 確認 `neuronpilot-6.0.5_x86_64.tar.gz` 位於專案根目錄

**模型轉換失敗：**
- 檢查PyTorch模型語法
- 確認輸入形狀格式：`(batch_size, ...dimensions)`
- 確保模型操作受到目標NPU支援

## 📞 技術支援

- **問題回報**：[GitHub Issues](https://github.com/R300-AI/MTK-NeuronPilot-API-docker/issues)
- **電子郵件**：support@r300.ai

---

**由 R300 AI 團隊用 ❤️ 製作**
