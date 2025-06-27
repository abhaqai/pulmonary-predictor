from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract features from form
            data = [
                float(request.form['age']),
                int(request.form['gender']),
                int(request.form['smoking']),
                int(request.form['finger_discoloration']),
                int(request.form['mental_stress']),
                int(request.form['pollution']),
                int(request.form['illness']),
                float(request.form['energy']),
                int(request.form['immune_weak']),
                int(request.form['breathing']),
                int(request.form['alcohol']),
                int(request.form['throat']),
                float(request.form['oxygen']),
                int(request.form['chest']),
                int(request.form['family']),
                int(request.form['smoke_history']),
                int(request.form['stress_immune']),
            ]

            # Make prediction
            prediction = model.predict([np.array(data)])
            result = 'Pulmonary Disease Detected (1)' if prediction[0] == 1 else 'No Disease Detected (0)'

            return render_template('index.html', prediction_text=result)

        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Main driver
if __name__ == '__main__':
    app.run(debug=True)
