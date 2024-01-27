from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
import tensorflow as tf
import librosa
import numpy as np 

#Load BLSTM Model
blstm_model_path = "../models/blstm3.h5"
blstm_model = load_model(blstm_model_path)

#Load CNN Model
cnn_model_path = "../models/cnn_audio.h5"
cnn_model = load_model(cnn_model_path)

#Load LSTM Model
lstm_model_path = "../models/audio_lstm.h5"
lstm_model = load_model(cnn_model_path)

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def predict_genre(model, audio_file_path, genre_mapping):

    # Load audio file
    signal, sample_rate = librosa.load(audio_file_path, sr=SAMPLE_RATE)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T

    # Reshape MFCCs to match model input shape
    mfcc = mfcc[:130, ...]  # Take only the first 130 MFCCs
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add extra dimensions


    # Predict using the model
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)

    # Map predicted index to genre label
    genre_label = genre_mapping[predicted_index[0]]
  

    return genre_label,prediction




VOICEDOR = "voices/"

 
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
 
@app.get('/file-upload', response_class=HTMLResponse)
def get_basic_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})
 
@app.post('/file-upload', response_class=HTMLResponse)
async def post_basic_form(request: Request, file: UploadFile = File(...)):
    print(f'Filename: {file.filename}')
    voice_addr = "voices/"+file.filename
     
    contents = await file.read()
     
    #save the file
    with open(f"{VOICEDOR}{file.filename}", "wb") as f:
        f.write(contents)
    # Genre mapping (update this according to your dataset)
    genre_mapping = {0: "fake", 1: "real"}


    # BLSTM prediction
    blstm_predict_class,blstm_raw_prediction = predict_genre(blstm_model, voice_addr, genre_mapping)
    if blstm_raw_prediction[0][0]>blstm_raw_prediction[0][1]:
        blstm_result_number = blstm_raw_prediction[0][0]
    else:
        blstm_result_number = blstm_raw_prediction[0][1]

    # CNN prediction
    cnn_predict_class,cnn_raw_prediction = predict_genre(cnn_model, voice_addr, genre_mapping)
    if cnn_raw_prediction[0][0]>cnn_raw_prediction[0][1]:
        cnn_result_number = cnn_raw_prediction[0][0]
    else:
        cnn_result_number = cnn_raw_prediction[0][1]

    #LSTM Prediction
    lstm_predict_class,lstm_raw_prediction = predict_genre(lstm_model, voice_addr, genre_mapping)
    if lstm_raw_prediction[0][0]>lstm_raw_prediction[0][1]:
        lstm_result_number = lstm_raw_prediction[0][0]
    else:
        lstm_result_number = lstm_raw_prediction[0][1]


    
    return templates.TemplateResponse("result.html", {"request": request, "file":file.filename, 
                                                      "blstm_predict_class":blstm_predict_class,
                                                      "blstm_result_number":blstm_result_number,
                                                      "cnn_predict_class":cnn_predict_class,
                                                      "cnn_result_number":cnn_result_number,
                                                      "lstm_predict_class":cnn_predict_class,
                                                      "lstm_result_number":cnn_result_number})


