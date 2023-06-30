from flask import Flask,render_template,request
import pickle

with open('diabetes_predictor','rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['post'])
def predictor():
    Pregnancies = request.form.get('Pregnancies')
    Glucose = request.form.get('Glucose')
    BloodPressure = request.form.get('BloodPressure')
    SkinThickness = request.form.get('SkinThickness')
    Insulin = request.form.get('Insulin')
    BMI = request.form.get('BMI')
    DiabetesPredictionFunction = request.form.get('DiabetesPredictionFunction')
    Age = request.form.get('Age')

    result = model.predict(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPredictionFunction, Age]])

    return str(result)

if __name__ == '__main__':
    app.run(debug=True)
