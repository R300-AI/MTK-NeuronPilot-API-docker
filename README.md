# How to Deploy a NeuronPilot Microservice on Azure App Service?

> [Requirements]
> * Docker Engine installed
> * An Azure Account
> * A subscription to purchase Azure products and services
> * An Azure **Container Registry** (recommended to be named `AIhubMicroServiceContainers`)

## Prepare the Docker Image on Workstation

1. **Download the Resources.**

    First, clone this repository to your workstation and download the NeuronPilot package.

    ```sh
    git clone https://github.com/R300-AI/neuronpilot-flask-server.git && cd neuronpilot-flask-server
    wget https://itriaihub.blob.core.windows.net/github-download-resources/repository/ITRI-AI-Hub/neuronpilot-6.0.5_x86_64.tar.gz
    ```

2. **Build the Docker image.**

    Next, build the Docker image for the NeuronPilot Flask service.
    ```sh
    docker build -f Dockerfile -t <registry_name>.azurecr.io/neuronpilot-converter .
    ```

3. **Test the NeuronPilot Flask Service.**

    Then, activate the Docker container with command `docker run -p 5000:80 <registry_name>.azurecr.io/neuronpilot-converter`. and use tools to verify it is working correctly.

    ```bash
    python tools.py
    ```
    `AIhubMicroServiceContainers`
    ```python
    from tools import Neuronpilot_WebAPI

    output_path = Neuronpilot_WebAPI(tflite_path = './uploads/yolov8n_float32.tflite', output_folder = './', url = 'http://localhost:5000/')
    print(f"File output path: {output_path}")
    ```
    

## Prepare the Azure Resources

1. **Login to Azure CLI and select your subscription.**

* Login Azure CLI and choice your subscription.
```
az config set core.enable_broker_on_windows=false
az login
```

2. **Set the registry name and login to Azure Container Registry.**

    Set the `registry_name` environment variable to your Azure Container Registry name and login.

    ```bash
    az acr login --name <registry_name>
    ```

3. **Build and tag the Docker image.**

    Build the Docker image using the Dockerfile and tag it with your Azure Container Registry name.

    ```bash
    docker tag neuronpilot-converter <registry_name>.azurecr.io/neuronpilot:latest
    ```

4. **Push the Docker image to Azure Container Registry.**

    Finally, push the Docker image to your Azure Container Registry.

    ```bash
    docker push <registry_name>.azurecr.io/neuronpilot-converter:latest
    ```
