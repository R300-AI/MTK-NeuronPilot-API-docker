import requests, os, argparse

def Neuronpilot_WebAPI(tflite_path, device, output_folder = './', url = 'http://localhost:80'):
    allowed_devices = ['mdla2.0', 'mdla3.0', 'vpu']
    assert device in allowed_devices, f"Device {device} is not allowed. Allowed devices are: {allowed_devices}"

    response = requests.post(url, 
                            files={'file': open(tflite_path,'rb')},
                            data={'device': device}
                            )
    print(f"Status: {response.status_code}")
    
    filename = response.headers.get('name')
    output_path = os.path.join(output_folder, filename)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    return output_path

def tflite_to_dla(tflite_name, device):
    dla_name = tflite_name.rstrip('.tflite') + '.dla'
    device_name = device.replace('.', '_')
    return dla_name.replace('.dla', f'_{device_name}.dla')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--url", default='https://app-aihub-neuronpilot.azurewebsites.net/', type=str)
    args = parser.parse_args()

    output_path = Neuronpilot_WebAPI(tflite_path = './uploads/yolov8n_float32.tflite', 
                                     device = 'mdla3.0',
                                     output_folder = './uploads',
                                     url = args.url)
    print(f"Converted file saved at: {output_path}")




