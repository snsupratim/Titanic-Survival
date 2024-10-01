from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


titanic_model_path = 'model01.pkl'  # Updated model name for Titanic survival prediction


with open(titanic_model_path, 'rb') as file:
    titanic_model = pickle.load(file)

app = Flask(__name__)



@app.route('/titanic_predict', methods=['POST'])
def titanic_predict():
    # Extract data from form for Titanic prediction
    age = request.form.get('Age', type=float)
    pclass = request.form.get('Pclass', type=int)
    sex_n = request.form.get('sex_n', type=int)  # Assuming sex_n is encoded as 0 or 1
    fare = request.form.get('Fare', type=float)

    # Validate inputs
    if age is None or pclass is None or sex_n is None or fare is None:
        return render_template('titanic.html', prediction_text='Invalid input. Please provide all fields.')

    final_features = np.array([[age, pclass, sex_n, fare]])
    
    # Make Titanic survival prediction
    prediction = titanic_model.predict(final_features)
    output = 'Survived' if prediction[0] == 1 else 'Did not survive'

    return render_template('titanic.html', prediction_text='Titanic Prediction: {}'.format(output))


@app.route('/')
def another_section():
    return render_template('titanic.html')

if __name__ == "__main__":
    app.run(debug=True)