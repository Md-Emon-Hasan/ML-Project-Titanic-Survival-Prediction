from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and data
pipe = pickle.load(open('models/pipe.pkl', 'rb'))
df = pickle.load(open('models/df.pkl', 'rb'))

# Convert 'Embarked' column to string to handle mixed types
df['Embarked'] = df['Embarked'].astype(str)

@app.route('/', methods=['GET'])
def index():
    # Sort the unique values
    Pclasss = sorted(df['Pclass'].unique())
    Sexs = sorted(df['Sex'].unique())
    Embarkeds = sorted(df['Embarked'].unique())

    return render_template('index.html',
                           Pclasss=Pclasss,
                           Sexs=Sexs,
                           Embarkeds=Embarkeds)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    Pclass = int(request.form['Pclass'])
    Sex = request.form['Sex']
    Age = float(request.form['Age'])
    SibSp = int(request.form['SibSp'])
    Parch = int(request.form['Parch'])
    Fare = float(request.form['Fare'])
    Embarked = request.form['Embarked']

    # Create a DataFrame from the input data for the model
    query = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]],
                         columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    # Predict survival
    prediction = pipe.predict(query)[0]

    # Sort the unique values again for consistent dropdown options
    Pclasss = sorted(df['Pclass'].unique())
    Sexs = sorted(df['Sex'].unique())
    Embarkeds = sorted(df['Embarked'].unique())

    # Render template with prediction and input values
    return render_template(
        'index.html',
        survived=prediction,
        Pclasss=Pclasss,
        Sexs=Sexs,
        Embarkeds=Embarkeds,
        Pclass=Pclass,
        Sex=Sex,
        Age=Age,
        SibSp=SibSp,
        Parch=Parch,
        Fare=Fare,
        Embarked=Embarked
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)