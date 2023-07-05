import os
import pickle
import numpy as np
from tqdm.notebook import tqdm

from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical

BASE_DIR='C:/Users/Nikola/datasets/flicker30k'
WORKING_DIR='C:/Users/Nikola/kaggle/working'

# load vgg16 model
model = VGG16()
# restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# extract features from image
features = {}

# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

#Load the captions data
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r', encoding='utf8') as f:
    next(f)
    captions_doc = f.read()

# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)

#Preprocess Text Data
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
            
# preprocess the text
clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

all_captions[:10]

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

# store features in pickle
pickle.dump(tokenizer, open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'wb'))

# load features from pickle
with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as t:
    tokenizer = pickle.load(t)

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
max_length = 35

#Train Test Split

image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

# startseq girl going into wooden building endseq
#        X                   y
# startseq                   girl
# startseq girl              going
# startseq girl going        into
# ...........
# startseq girl going into wooden building      endseq

# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    # store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0

batch_size = 32
steps = len(train) // batch_size

from keras.models import load_model
# load the model vgg16_lstm256 Epoch20*10
model = load_model('C:/Users/Nikola/kaggle/working/best_model.h5')

#Generate Captions for the Image
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text

def get_feature(img_name):
# for img_name in tqdm(os.listdir(directory)):
    # load the image from file
    # load vgg16 model
    model = VGG16()
    # restructure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    img_path = 'C:/Users/Nikola/Desktop/denemeImg' + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    feature = model.predict(image, verbose=0)
    # get image ID
    image_id = img_name.split('.')[0]
    #store feature
    features[image_id] = feature

from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join('C:/Users/Nikola/Desktop', "denemeImg", image_name)
    image = Image.open(img_path)
    get_feature(image_name)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    #print('--------------------Predicted--------------------')
    #print(y_pred)    
    #plt.imshow(image)
    return y_pred

from keras_preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

def detect_object(image_name):
    detectModel = VGG16(weights='imagenet')

    img_path = os.path.join('C:/Users/Nikola/Desktop', "denemeImg", image_name)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = detectModel.predict(x)
    #print('Predicted:', decode_predictions(preds, top=3)[0])
    tuppleDetectionList = decode_predictions(preds, top=5)[0]

    detection_list = []
    for detection in tuppleDetectionList:  
        desc = detection[1]
        detection_list.append(desc)

    return detection_list

from flask import Flask,jsonify,request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# GET isteği için endpoint
@app.route("/get_image_caption")
def get_image_caption():
    image_name = request.args.get('image_name')
    return jsonify(generate_caption(image_name).replace('endseq','').replace('startseq',''))

@app.route("/get_object_detection")
def get_object_detection():
    image_name = request.args.get('image_name')
    return jsonify(detect_object(image_name))

if __name__ == "__main__":
    app.run(debug=True, port=5999)


#http://127.0.0.1:5000/get_object_detection?image_name=test.jpg
#http://127.0.0.1:5000/get_image_caption?image_name=test.jpg