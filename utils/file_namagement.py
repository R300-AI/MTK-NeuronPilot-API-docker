import datetime, random

def random_seed():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_number = random.randint(1000, 9999)
    return f"{timestamp}_{random_number}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'tflite'}