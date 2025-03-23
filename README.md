# How to Deploy a NeuronPilot Converter Service?

In this document, you will learn how to build a NeuronPilot Converter on a local server workstation (x86_64) or on Azure cloud to provide a Restful API for clients to convert `.tflite` models to `.dla` models.

### Requirements
> * A x86_64 Workstation with **VS Code** and **Docker Engine** installed.
> 
> :point_down:Cloud-based Service only
> * A valid **Azure Service** account
> * A **Subscription** with purchased Azure Services
> * A **Container Registry** Resource for storing your docker image (recommended to be named `aihubmicroservice`)


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
    docker tag neuronpilot-converter <registry_name>.azurecr.io/neuronpilot-converter
    ```

    <img src="https://github.com/R300-AI/neuronpilot-flask-server/blob/main/images/registry_portal.png" width="900">

4. **Push the Docker image to Azure Container Registry.**

    Finally, push the Docker image to your Azure Container Registry.

    ```bash
    docker push <registry_name>.azurecr.io/neuronpilot-converter
    ```


## Deploy Image to Azure App Service (Must use VS Code)

1. From the Docker extension in VS Code, select **Deploy Image to Azure App Service**.
    <img src="https://github.com/R300-AI/neuronpilot-flask-server/blob/main/images/deploy_service_tutorial/step1.png" width="900"><br>

2. Choose your **Subscription**.
    <img src="https://github.com/R300-AI/neuronpilot-flask-server/blob/main/images/deploy_service_tutorial/step2.png" width="720"><br>

3. Provide a name for your App Service. For example, `<app-service-name>` could be `app-aihub-neuronpilot`.
    <img src="https://github.com/R300-AI/neuronpilot-flask-server/blob/main/images/deploy_service_tutorial/step3.png" width="720"><br>

4. Select your **Resource Group**.
    <img src="https://github.com/R300-AI/neuronpilot-flask-server/blob/main/images/deploy_service_tutorial/step4.png" width="720"><br>

5. Choose **Create new App Service plan**.
    <img src="https://github.com/R300-AI/neuronpilot-flask-server/blob/main/images/deploy_service_tutorial/step5.png" width="720"><br>

6. Enter a name for your **App Service Plan**.
    <img src="https://github.com/R300-AI/neuronpilot-flask-server/blob/main/images/deploy_service_tutorial/step6.png" width="720"><br>

7. Select an appropriate pricing tier.
    <img src="https://github.com/R300-AI/neuronpilot-flask-server/blob/main/images/deploy_service_tutorial/step7.png" width="720"><br>

8. Disable redundancy.
    <img src="https://github.com/R300-AI/neuronpilot-flask-server/blob/main/images/deploy_service_tutorial/step8.png" width="720"><br>

9. Wait for the success notification to appear in the bottom right corner of VS Code.
    <img src="https://github.com/R300-AI/neuronpilot-flask-server/blob/main/images/deploy_service_tutorial/step9.png" width="600">
    
10. Finally, test the deployed service using the provided Python API.
    ```python
    from tools import Neuronpilot_WebAPI

    output_path = Neuronpilot_WebAPI(tflite_path = './uploads/yolov8n_float32.tflite', output_folder = './', url = '[http://localhost:5000/](https://<app-service-name>.azurewebsites.net/)')
    print(f"Converted file saved at: {output_path}")
    ```
