from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the preprocessing steps and the model
label_encoders = joblib.load('label_encoders.pkl')
minmax_scalers = joblib.load('scaler.pkl')
model = joblib.load('rf_bestest_model.pkl')

selected_features = ['RECORD_TYPE', 'STATE', 'GROUP_SIZE', 'HOMEOWNER', 'CAR_AGE', 'CAR_VALUE', 'RISK_FACTOR', 'MARRIED_COUPLE', 'AGE_OLDEST', 'C_PREVIOUS', 'DURATION_PREVIOUS', 'A','B', 'C', 'E', 'G']
categorical_columns = ['STATE', 'CAR_VALUE']
numerical_columns = ['CAR_AGE', 'AGE_OLDEST']

def preprocess_input(data):
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])

    # Initialize a dictionary to hold preprocessed values
    preprocessed_data = {}

    # Convert categorical variables using label_encoders
    for col in categorical_columns:
        if col in data:
            preprocessed_data[col] = label_encoders[col].transform([data[col]])[0]

    df[numerical_columns] = minmax_scalers.transform(df[numerical_columns])

    # Create DataFrame from preprocessed_data
    df = pd.DataFrame([preprocessed_data])

    # Ensure all selected features are present, filling with zeros if missing
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0

    return df[selected_features]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Preprocess the input data
    preprocessed_data = preprocess_input(data)

    # Make prediction using the model
    prediction = model.predict(preprocessed_data)

    # Round prediction to 2 decimal places
    rounded_prediction = round(prediction[0], 2)

    # Return rounded prediction as JSON response
    return jsonify({'prediction': rounded_prediction})

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Preprocess the input data
    preprocessed_data = preprocess_input(data)

    # Make prediction using the model
    prediction = model.predict(preprocessed_data)

    # Round prediction to 2 decimal places
    rounded_prediction = round(prediction[0], 2)

    # Return rounded prediction as JSON response
    return jsonify({'prediction': rounded_prediction})


if __name__ == '__main__':
    app.run(debug=True)
