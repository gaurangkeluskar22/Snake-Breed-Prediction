from __future__ import division, print_function
import sys
import os
import glob
import re
import tensorflow as tf
import tensorflow_hub as hub
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# Define a flask app
app = Flask(__name__)
unique_labels=['agkistrodon-contortrix', 'agkistrodon-piscivorus',
       'coluber-constrictor', 'crotalus-atrox', 'crotalus-horridus',
       'crotalus-ruber', 'crotalus-scutulatus', 'crotalus-viridis',
       'diadophis-punctatus', 'haldea-striatula', 'heterodon-platirhinos',
       'lampropeltis-californiae', 'lampropeltis-triangulum',
       'masticophis-flagellum', 'natrix-natrix', 'nerodia-erythrogaster',
       'nerodia-fasciata', 'nerodia-rhombifer', 'nerodia-sipedon',
       'opheodrys-aestivus', 'pantherophis-alleghaniensis',
       'pantherophis-emoryi', 'pantherophis-guttatus',
       'pantherophis-obsoletus', 'pantherophis-spiloides',
       'pantherophis-vulpinus', 'pituophis-catenifer',
       'rhinocheilus-lecontei', 'storeria-dekayi',
       'storeria-occipitomaculata', 'thamnophis-elegans',
       'thamnophis-marcianus', 'thamnophis-proximus', 'thamnophis-radix',
       'thamnophis-sirtalis']
       
def load_model(model_path):
  print(f'Loading model from :{model_path}')
  model=tf.keras.models.load_model(model_path,
                                   custom_objects={
                                       'KerasLayer':hub.KerasLayer
                                   })
  return model

def get_pred_label(prediction_probability,unique_labels):
  return unique_labels[np.argmax(prediction_probability)]  

def get_pred_get_pred(custom_preds):
    custom_pred_label=[get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
    custom_pred_label

def get_image_label(image_path,label):
  image=preprocess_image(image_path)
  return image,label

# define image size
IMG_SIZE=224
BATCH_SIZE=32

def preprocess_image(image_path,img_size=224):
  image=tf.io.read_file(image_path)
  image=tf.image.decode_jpeg(image,channels=3)
  image=tf.image.convert_image_dtype(image,tf.float32)
  image = tf.image.resize(image,size=[IMG_SIZE,IMG_SIZE])
  return image


def create_data_batches(x,batch_size=32):
    print('Creating test data branches....')
    x=[x]
    data=tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch=data.map(preprocess_image).batch(BATCH_SIZE)
    return data_batch


def model_predict(custom_images_path,loaded_full_model):
    custom_data=create_data_batches(custom_images_path)
    custom_preds = loaded_full_model.predict(custom_data)
    custom_pred_label=[get_pred_label(custom_preds[i],unique_labels) for i in range(len(custom_preds))]
    return custom_pred_label


loaded_full_model=load_model('model/snake-model-second.h5')
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        #Make prediction
        result = model_predict(file_path,loaded_full_model)            
        return result[0]
    return None


if __name__ == '__main__':
    app.run(debug=True)



