# NeuronPilot-Flask-Service-docker
* Download NeuronPilot Package
```
wget https://itriaihub.blob.core.windows.net/github-download-resources/repository/ITRI-AI-Hub/neuronpilot-6.0.5_x86_64.tar.gz
```
* Login Azure CLI with subscription
```
az account clear
az config set core.enable_broker_on_windows=false
az login
```

https://learn.microsoft.com/en-us/azure/app-service/quickstart-custom-container?tabs=python&pivots=container-linux-azure-portal
