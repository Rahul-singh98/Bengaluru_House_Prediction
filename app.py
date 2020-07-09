# Importing essential libraries
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

# Load the model
H_filename = 'House_Price_Prediction_Bengluru/Bengluru_house_prediction.pkl'
H_model = pickle.load(open(H_filename, 'rb'))

X = pd.read_csv('House_Price_Prediction_Bengluru/_Data.csv' , index_col = 0)

app = Flask(__name__)

def prediction(location, bhk, bath, balcony, sqft, area_type, availability):

    loc_index, area_index, avail_index = -1,-1,-1
        
    if location!='other':
        loc_index = int(np.where(X.columns==location)[0][0])
    
    if area_type!='Super built-up  Area':
        area_index = np.where(X.columns==area_type)[0][0]
        
    if availability!='Not Ready':        
        avail_index = np.where(X.columns==availability)[0][0]
            
    x = np.zeros(len(X.columns))
    x[0] = bath
    x[1] = balcony
    x[2] = bhk
    x[3] = sqft
    
    if loc_index >= 0:
        x[loc_index] = 1
    if area_index >= 0:
        x[area_index] = 1
    if avail_index >= 0:
        x[avail_index] = 1
        
    return H_model.predict([x])[0]

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/House_predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        loc = str(request.form['Location'])
        bhk = int(request.form['BHK'])
        bath = int(request.form['Bathroom'])
        balc = int(request.form['Balcony'])
        sqft = int(request.form['Square Fit'])
        a_type = str(request.form['Area Type'])
        avail = str(request.form['Availability'])
        
        my_prediction = prediction(loc, bhk, bath, balc, sqft, a_type, avail)
        
        return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
