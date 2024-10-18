from flask import Flask, request, jsonify, send_from_directory, url_for
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
from gtts import gTTS
import io
import os
import json
import cv2
import numpy as np
from multiprocessing import Value
from flask import render_template

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_length = 30
gen_kwargs = {"max_length": max_length}
# declare counter variable
counter = Value('i', 0)

def save_img(img):
	with counter.get_lock():
		counter.value += 1
		count = counter.value
	img_dir = "esp32_imgs"
	if not os.path.isdir(img_dir):
		os.mkdir(img_dir)
	cv2.imwrite(os.path.join(img_dir,"img_"+str(count)+".jpg"), img)
	# print("Image Saved", end="\n") # debug

def load_model_and_processor():
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
    model.to(device)
    return model, processor

model, processor = load_model_and_processor()

def generate_caption(image, model, processor):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    generated_ids = model.generate(pixel_values=pixel_values, **gen_kwargs)
    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return captions[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST','GET'])
def upload():
	received = request
	img = None
	if received.files:
		print(received.files['imageFile'])
		# convert string of image data to uint8
		file  = received.files['imageFile']
		nparr = np.fromstring(file.read(), np.uint8)
		# decode image
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		save_img(img)
		captioning = predict()
                
		return "[SUCCESS] Image Received " + captioning, 201
	else:
		return "[FAILED] Image Not Received", 204



#@app.route('/predict', methods=['POST'])
# def predict():
#     esp32_imgs_dir = 'esp32_imgs'
#     files = os.listdir(esp32_imgs_dir)
    
#     if not files:
#         return jsonify({'error': 'No files in directory'})
    
#     image_file = None
#     for file in files:
#         if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
#             image_file = os.path.join(esp32_imgs_dir, file)
#             break
    
#     if image_file is None:
#         return jsonify({'error': 'No image file found in directory'})
    
#     image = Image.open(image_file)
#     caption = generate_caption(image, model, processor)
    
#     tts = gTTS(caption)
#     tts.save('static/caption.mp3')
    
#     return caption

 
@app.route('/predict', methods=['POST'])
def predict():
    esp32_imgs_dir = 'esp32_imgs'
    files = os.listdir(esp32_imgs_dir)
    
    if not files:
        return jsonify({'error': 'No files in directory'})
    
    image_file = None
    for file in files:
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            image_file = os.path.join(esp32_imgs_dir, file)
            break
    
    if image_file is None:
        return jsonify({'error': 'No image file found in directory'})
    
    try:
        image = Image.open(image_file)
        caption = generate_caption(image, model, processor)
        
        tts = gTTS(caption)
        tts.save('static/caption.mp3')
        
        response = jsonify({'caption': caption})
    except Exception as e:
        response = jsonify({'error': str(e)})
    
    # Cleanup: delete all files in the esp32_imgs directory
    for file in files:
        file_path = os.path.join(esp32_imgs_dir, file)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    
    return response

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
