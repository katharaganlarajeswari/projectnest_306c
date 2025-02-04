from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    Item_Weight = float(request.form['Item_Weight'])
    Item_MRP = float(request.form['Item_MRP'])

    # Prepare the input features
    features = np.array([[Item_Weight, Item_MRP]])

    # Make the prediction using the loaded model
    prediction = model.predict(features)

    # Render the result.html template with the prediction result
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
