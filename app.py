from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the model
model_path = 'student_marks_model.pkl'  # Correct path separator
with open(model_path, 'rb') as f:  # Use 'rb' for loading the pickle file
    model = pickle.load(f)

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract the inputs from the form in the new order
    time_study = float(request.form['time_study'])
    number_courses = int(request.form['number_courses'])
    time_division = float(request.form['time_division'])

    # Prepare the input for the model
    final_features = np.array([[time_study, number_courses, time_division]])

    # Make a prediction
    prediction = model.predict(final_features)

    # Render the result on the page
    return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)
