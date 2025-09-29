import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from model import predict_emotion # <-- MODIFICATION: Import the final prediction function

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            prediction_result = 'No file part in the request.'
            return render_template('index.html', prediction=prediction_result)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            prediction_result = 'No file selected. Please upload an audio file.'
            return render_template('index.html', prediction=prediction_result)

        if file:
            # Sanitize the filename for security
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # --- MODIFICATION: Call the actual model for prediction ---
            # The placeholder logic is removed.
            try:
                predicted_emotion = predict_emotion(filepath)
                prediction_result = f'Detected Emotion: {predicted_emotion.capitalize()}'
            except Exception as e:
                prediction_result = f'Error processing file: {e}'

    # Render the page with the result (or None if it's a GET request)
    return render_template('index.html', prediction=prediction_result)


if __name__ == '__main__':
    app.run(debug=True)
