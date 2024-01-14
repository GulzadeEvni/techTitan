from flask import Flask, request, render_template
from keras.models import load_model
import librosa
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify


app = Flask(__name__)

model = load_model('my_model.h5')
with open('pkl.pkl', 'rb') as file_pi:
    history = pickle.load(file_pi)

def detect_fake(sound_file):
    sound_signal, sample_rate = librosa.load(sound_file, res_type="kaiser_fast")
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
    mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)
    result_array = model.predict(mfccs_features_scaled)
    result_classes = ["FAKE", "REAL"]
    result = np.argmax(result_array[0])
    return result_classes[result]

# Ana sayfa
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = 'uploads/' + filename
            file.save(file_path)
            result = detect_fake(file_path)
            return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)
