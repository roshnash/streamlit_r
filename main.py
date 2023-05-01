import pickle
import librosa
import numpy as np
from os import remove
import streamlit as st
from os.path import join
from keras.models import load_model

st.title("Classifying Bird Calls of Endangered Bird Species of Kerala")

SAMPLE_RATE = 22050


def upload_and_save(infile):
    try:
        with open(join("uploaded", infile.name), "wb") as f:
            f.write(infile.getbuffer())
        return 1
    except:
        return 0


def wav_loader(fname):
    raw_audio, Fs = librosa.load(fname, sr=SAMPLE_RATE)
    return np.array(raw_audio).reshape(1, -1)


def load_trained_model():
    best_model_path = "best_dnn_fft_model.hdf5"
    model = load_model(join("static", best_model_path))
    return model


def load_class_map():
    with open(join("static", "class_map.pkl"), "rb") as f:
        class_map = pickle.load(f)
    class_map_inv = {v: k for k, v in class_map.items()}
    return class_map_inv


infile = st.file_uploader("Upload Bird Call Audio")

if infile is not None:
    if upload_and_save(infile):
        x = wav_loader(join("uploaded", infile.name))
        xfft = np.abs(np.fft.fft(x))

        model = load_trained_model()
        y_pred_probs = model.predict(xfft)[0]

        class_map_inv = load_class_map()
        y_label_indx = np.argmax(y_pred_probs)
        species_predicted = class_map_inv[y_label_indx]

        remove(join("uploaded", infile.name))

        st.text("Species of Uploaded Bird Call is:")
        st.text(species_predicted)
