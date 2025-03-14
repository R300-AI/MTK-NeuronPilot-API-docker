FROM ubuntu:22.04

COPY . /app
WORKDIR /app

RUN apt update && apt install -y bash
RUN apt install -y build-essential 
RUN apt install -y libncurses5 libstdc++6 libc++1
RUN apt install -y python3-pip

RUN tar zxvf neuronpilot-6.0.5_x86_64.tar.gz -C ./
RUN bash -c "./neuronpilot-6.0.5/neuron_sdk/host/bin/ncc-tflite --version"
RUN bash -c "pip install -r requirements.txt"

EXPOSE 80

CMD ["python3" ,"-u" , "app.py"]