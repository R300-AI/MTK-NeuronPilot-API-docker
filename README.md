# How to Deploy a NeuronPilot Converter Service?

In this document, you will learn how to build a NeuronPilot Converter on a local server workstation (x86_64) or on Azure cloud to provide a Restful API for clients to convert `.tflite` models to `.dla` models.

### Requirements
> * A x86_64 Workstation with **VS Code** and **Docker Engine** installed.

## Prepare the Docker Image on Workstation

1. **Download the Resources.**

    First, clone this repository to your workstation and download the NeuronPilot package.

    ```sh
    git clone https://github.com/R300-AI/neuronpilot-flask-server.git && cd neuronpilot-flask-server
    wget https://githubfileshare.blob.core.windows.net/repo/neuronpilot-flask-server/neuronpilot-6.0.5_x86_64.tar.gz
    ```

2. **Build the Docker image.**

    Next, build the Docker image for the NeuronPilot Converter service.

    ```sh
    docker build -t neuronpilot-converter .
    ```

3. **Test the NeuronPilot Service Locally.**

    Then, activate the Docker container with the command `docker run -p 5000:80 neuronpilot-converter` and use tools to verify it is working correctly.

    ```bash
    python tools.py --url http://localhost:5000
    ```

4. Finally, test the deployed service using the provided Python API.
    ```python
    from tools import Neuronpilot_WebAPI

    output_path = Neuronpilot_WebAPI(tflite_path = './uploads/yolov8n_float32.tflite', output_folder = './', url = 'http://localhost:5000')
    print(f"Converted file saved at: {output_path}")
    ```
