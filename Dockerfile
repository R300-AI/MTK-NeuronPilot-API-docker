FROM ubuntu:22.04

COPY . /app
WORKDIR /app

RUN apt update && apt install -y bash
RUN apt install -y build-essential 
RUN apt install -y libncurses5 libstdc++6 libc++1
RUN apt install -y python3-pip wget

# Download and extract NeuronPilot SDK
RUN if [ ! -f "neuronpilot-6.0.5_x86_64.tar.gz" ]; then \
        echo "NeuronPilot SDK not found locally, downloading..."; \
        wget https://itriaihub.blob.core.windows.net/github-download-resources/repository/ITRI-AI-Hub/neuronpilot-6.0.5_x86_64.tar.gz; \
    else \
        echo "Found existing NeuronPilot SDK file, skipping download"; \
    fi
RUN tar zxvf neuronpilot-6.0.5_x86_64.tar.gz -C ./
RUN bash -c "./neuronpilot-6.0.5/neuron_sdk/host/bin/ncc-tflite --version"
RUN bash -c "pip install --timeout 1000 --retries 5 -r requirements.txt"
ENV PATH="/root/.local/bin:$PATH"

EXPOSE 80

CMD ["python3" ,"-u" , "app.py"]
#CMD ["/bin/bash"]