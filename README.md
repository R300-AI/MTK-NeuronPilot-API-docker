# How to Deploy a NeuronPilot Converter Service?

In this document, you will learn how to build a NeuronPilot Converter on a local server workstation (x86_64) or on Azure cloud to provide a Restful API for clients to convert `.tflite` models to `.dla` models.

### Requirements
> * A x86_64 Workstation with **Docker Engine** installed.
> 
> :point_down:Cloud-based Service only
> * A valid **Azure Service** account
> * A **Subscription** with purchased Azure Services
> * A **Container Registry** Resource for storing your docker image (recommended to be named `AIhubMicroServiceContainers`)


## Prepare the Docker Image on Workstation

1. **Download the Resources.**

    First, clone this repository to your workstation and download the NeuronPilot package.

    ```sh
    git clone https://github.com/R300-AI/neuronpilot-flask-server.git && cd neuronpilot-flask-server
    wget https://itriaihub.blob.core.windows.net/github-download-resources/repository/ITRI-AI-Hub/neuronpilot-6.0.5_x86_64.tar.gz
    ```

2. **Build the Docker image.**

    Next, build the Docker image for the NeuronPilot Converter service.

    ```sh
    docker build -t neuronpilot-converter .
    ```

3. **Test the NeuronPilot Service Locally.**

    Then, activate the Docker container with the command `docker run -p 5000:80 neuronpilot-converter` and use tools to verify it is working correctly.

    ```bash
    python tools.py
    ```
    * You can also use the provided Python API in the file to call the SDK.
    ```python
    from tools import Neuronpilot_WebAPI

    output_path = Neuronpilot_WebAPI(tflite_path = './uploads/yolov8n_float32.tflite', output_folder = './', url = 'http://localhost:5000/')
    print(f"Converted file saved at: {output_path}")
    ```


## Prepare the Azure Container Registry

1. **Login to Azure CLI and select your subscription.**

    Login to Azure CLI and choose your subscription.

    ```
    az config set core.enable_broker_on_windows=false
    az login
    ```

2. **Set the registry name and login to Azure Container Registry.**

    Set your Azure Container Registry name to `registry_name` and login to the Registry.

    ```bash
    az acr login --name <registry_name>
    ```

3. **Build and tag the Docker image.**

    Tag the Docker image with your Azure Container Registry name. For more details, you can refer to the **"Push an image" tutorial** at the Container Registry's Portal.
    
    ```bash
    docker tag neuronpilot-converter <registry_name>.azurecr.io/neuronpilot
    ```
    ![Registry Portal](https://github.com/R300-AI/neuronpilot-flask-server/blob/main/static/images/registry_portal.png)


4. **Push the Docker image to Azure Container Registry.**

    Finally, push the Docker image to your Azure Container Registry.

    ```bash
    docker push <registry_name>.azurecr.io/neuronpilot-converter
    ```
