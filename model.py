# --- INSTALL DEPENDENCIES ---
# Run this in your terminal: pip install librosa tensorflow scikit-learn resampy

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, Dense
from tensorflow.keras.utils import to_categorical

# --- PATHS ---
DATASET_PATH = "./ravdess"  # Path to your locally downloaded RAVDESS folder
MODEL_PATH = "./voicevibes_cnn_model.h5"

# --- EMOTIONS MAP ---
EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# --- FEATURE EXTRACTION ---
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# --- LOAD DATASET ---
def load_dataset(dataset_path):
    features, labels = [], []
    actors = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    for actor in actors:
        actor_path = os.path.join(dataset_path, actor)
        for file_name in os.listdir(actor_path):
            try:
                emotion_code = file_name.split('-')[2]
                emotion = EMOTIONS[emotion_code]
                file_path = os.path.join(actor_path, file_name)
                mfccs = extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(emotion)
            except Exception as e:
                continue
    features = np.array(features)
    labels = np.array(labels)
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    return features, labels

# --- BUILD CNN MODEL ---
def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(256, 5, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, padding='same', activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(len(EMOTIONS), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- TRAIN MODEL ---
def train_model():
    features, labels = load_dataset(DATASET_PATH)
    if features.shape[0] == 0:
        print("No features extracted. Check your dataset path and audio files.")
        return

    # Encode labels
    label_to_int = {label: i for i, label in enumerate(np.unique(labels))}
    labels_int = np.array([label_to_int[label] for label in labels])
    labels_categorical = to_categorical(labels_int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    model.summary()

    print("Training model...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
    model.save(MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")

# --- PREDICTION ---
def load_trained_model():
    return load_model(MODEL_PATH)

def predict_emotion(audio_file_path):
    model = load_trained_model()
    features = extract_features(audio_file_path)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction, axis=1)[0]
    unique_emotions = sorted(list(EMOTIONS.values()))
    return unique_emotions[predicted_index]

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Train the CNN
    train_model()

    # Test prediction on a single file
    test_actor_folder = os.path.join(DATASET_PATH, "Actor_01")
    test_file = os.path.join(test_actor_folder, os.listdir(test_actor_folder)[0])
    print("Testing on:", test_file)
    print("Predicted Emotion:", predict_emotion(test_file))
