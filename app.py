from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('medical_data.csv')

# Step 1: Clean column names (optional)
data.columns = data.columns.str.strip()

# Step 2: Check if 'disease' is present as the target column
target_column = 'disease'  # Change this to the correct column name if necessary
if target_column not in data.columns:
    raise KeyError(f"'{target_column}' column not found in the dataset. Please check the column name.")

# Prepare features (X) and target (y)
X = data.drop(columns=[target_column])
y = data[target_column]

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Save the trained model as a .pkl file (to be used later)
joblib.dump(model, 'disease_predictor_model.pkl')

# Step 6: Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Step 7: Prediction route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json()
        age = data['age']
        blood_pressure = data['bloodPressure']
        cholesterol = data['cholesterol']

        # Prepare the input data for the model
        input_data = pd.DataFrame([[age, blood_pressure, cholesterol]], columns=["age", "blood_pressure", "cholesterol"])

        # Load the saved model and make a prediction
        model = joblib.load('disease_predictor_model.pkl')
        prediction = model.predict(input_data)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
