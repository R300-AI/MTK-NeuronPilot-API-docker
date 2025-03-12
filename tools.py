import requests, os

def Neuronpilot_WebAPI(tflite_path, output_folder = './', url = 'http://localhost:5001/'):
    response = requests.post(url, files={'file': open(tflite_path,'rb')})
    print(f"Status: {response.status_code}")
    
    filename = response.headers.get('name')
    output_path = os.path.join(output_folder, filename)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    return output_path

if __name__ == '__main__':
    output_path = Neuronpilot_WebAPI(tflite_path = './uploads/yolov8n_float32.tflite', output_folder = './')
    print(f"File output path: {output_path}")




