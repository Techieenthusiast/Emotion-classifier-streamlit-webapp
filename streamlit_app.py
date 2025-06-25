import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import os

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# Custom Attention layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        alpha = K.softmax(e, axis=1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5", custom_objects={"Attention": Attention})

model = load_model()
emotion_classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Feature extraction
def extract_features(file_path, max_pad_len=200):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs.T

# Streamlit UI
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload one or more `.wav` files to predict emotions:")

uploaded_files = st.file_uploader("Upload WAV files", type=["wav"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.read())

        st.audio(file, format="audio/wav")
        st.markdown(f"**Processing file:** `{file.name}`")

        try:
            features = extract_features(file.name)
            features = features[np.newaxis, ..., np.newaxis]
            prediction = model.predict(features, verbose=0)
            predicted_emotion = emotion_classes[np.argmax(prediction)]

            st.success(f"🎯 Predicted Emotion: {predicted_emotion}")

        except Exception as e:
            st.error(f"❌ Error processing `{file.name}`: {e}")

        # Clean up temporary file
        os.remove(file.name)

