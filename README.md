# VoiceVibes

**VoiceVibes** is a Flask-based web application that leverages a trained CNN model to perform voice-related predictions. This project demonstrates how to integrate machine learning models with a web interface for interactive user experience.

---

## 📝 Project Structure

```

VoiceVibes/
├── app.py                 # Flask application
├── model.py               # Model loading and prediction logic
├── requirements.txt       # Python dependencies
├── voicevibes_cnn_model.h5 # Pre-trained CNN model
├── templates/
│   └── index.html         # HTML template for UI
└── static/
    └── style.css          # Style template for UI

````

---

## ⚡ Features

- Load a pre-trained CNN model for voice prediction.
- User-friendly web interface via Flask.
- Easy to extend and integrate with other voice or ML applications.

---

## 💻 Installation

1. **Clone the repository**
```bash
git clone https://github.com/HariN999/VoiceVibes.git
cd VoiceVibes
````

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```
4. **Download / Generate the Model**
Before running the Flask app, you need the model file. Run:

```bash
python model.py 
```
💡 Tip: It is recommended to use Google Colab for generating or training the model, especially if your local machine has limited resources. Colab provides free GPU support and ensures faster model creation.

This will either download the pre-trained CNN model or generate it locally, saving it as voicevibes_cnn_model.h5 in the project folder.

## 🚀 Running the Application

```bash
python app.py
```

* Open your browser and navigate to `http://127.0.0.1:5000` to access the web interface.

---

## 📁 Usage

1. Open the app in your browser.
2. Upload or provide input as required by the model.
3. Click **Predict** to get results based on the CNN model.

---

## ⚙️ Customization

* Modify `model.py` to load different models or change prediction logic.
* Update `templates/index.html` for a different UI layout.
* Add new routes in `app.py` for additional functionality.

---

## 🧰 Dependencies

* Python 3.8+
* Flask
* TensorFlow/Keras
* Numpy

Install all required packages via:

```bash
pip install -r requirements.txt
```

## 🔗 Author

**Hariharan Narlakanti**
[GitHub](https://github.com/HariN999) | [LinkedIn](https://www.linkedin.com/in/narlakanti-hariharan)

```
```
