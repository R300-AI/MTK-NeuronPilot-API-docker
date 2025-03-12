# How to Deploy a NeuronPilot Microservice on Azure App Service?
## Prepare the Docker Image on Workstation
> [Requirements]
> * Docker Engine

1. **Download the NeuronPilot package and move it to the `neuronpilot-flask-server` folder.**

    First, clone this repository to your workstation and download the NeuronPilot package.

    ```sh
    git clone https://github.com/R300-AI/neuronpilot-flask-server.git && cd neuronpilot-flask-server
    
    wget https://itriaihub.blob.core.windows.net/github-download-resources/repository/ITRI-AI-Hub/neuronpilot-6.0.5_x86_64.tar.gz
    mv neuronpilot-6.0.5_x86_64.tar.gz <path-to-this-repository>
    ```

    Replace `<path-to-this-repository>` with the actual path to this repository.

2. **Build the Docker image for the NeuronPilot Flask service.**

    Next, build the Docker image for the NeuronPilot Flask service.
    ```sh
    docker build -t neuronpilot-converter .
    
    ```

3. **Run the Docker container and test the NeuronPilot Flask service.**

    Then, run the Docker container and test the NeuronPilot Flask service to ensure it is working correctly.


    ```bash
    docker run -p 5000:80 neuronpilot-converter
    ```
    ```python
    from tools import Neuronpilot_WebAPI

    output_path = Neuronpilot_WebAPI(tflite_path = './uploads/yolov8n_float32.tflite', output_folder = './', url = 'http://localhost:5000/')
    print(f"File output path: {output_path}")
    ```
    

## Prepare Azure Resources
* Login Azure CLI with subscription
```
az account clear
az config set core.enable_broker_on_windows=false
az login

az acr login --name neuronpilot

docker pull mcr.microsoft.com/mcr/hello-world
docker tag mcr.microsoft.com/mcr/hello-world neuronpilot.azurecr.io/samples/hello-world

docker push neuronpilot.azurecr.io/samples/hello-world
```

https://learn.microsoft.com/en-us/azure/app-service/quickstart-custom-container?tabs=python&pivots=container-linux-azure-portal
