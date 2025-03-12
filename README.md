# NeuronPilot-Flask-Service-docker
## Prepare Docker Container
* Download NeuronPilot package and move it to `neuronpilot-flask-server` folder.
```
$ wget https://itriaihub.blob.core.windows.net/github-download-resources/repository/ITRI-AI-Hub/neuronpilot-6.0.5_x86_64.tar.gz
$ mv neuronpilot-6.0.5_x86_64.tar.gz <path-to-this-repository>
```

* Build Docker Container
```
$ docker build -t my-flask-app .

# Test on your browser `localhost:5000`
$ docker run -e PYTHONUNBUFFERED=1 -p 5001:80 my-flask-app

# Test inside container
$ docker run -p 5001:80 -it my-flask-app
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
