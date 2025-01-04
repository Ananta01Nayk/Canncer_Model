from flask import Flask,render_template,request,url_for
import pickle
import pandas as pd
import numpy as npcls
import joblib

model = pickle.load(open('pipe.pkl','rb'))
mocel_1 =joblib.load('breast_cancer.joblib')

app =  Flask(__name__)

@app.route("/")

def main():
    return render_template("home.html")

@app.route("/forms")
def forms():
    return render_template("Helthform.html")

@app.route("/About")
def About():
    return render_template("about.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/Service")
def Service():
    return render_template("service.html")

@app.route("/Contact")
def Contact():
    return render_template("Contact.html")

@app.route("/learn")
def learn():
    return render_template("learn.html")

@app.route("/breast_cancer")
def breast_cancer():
    return render_template("breast_cancer.html")

@app.route("/about")
def about():
    return render_template('about.html')

@app.route('/predict',methods=['POST'])
def predict():
    GENDER = request.form['GENDER']
    AGE = request.form['AGE']
    SMOKING=request.form['SMOKING']
    YELLOW_FINGERS= request.form['YELLOW_FINGERS']
    ANXIETY	= request.form['ANXIETY']
    PEER_PRESSURE = request.form['PEER_PRESSURE']
    CHRONIC_DISEASE = request.form['CHRONIC_DISEASE']

    FATIGUE	= request.form['FATIGUE']
    ALLERGY = request.form['ALLERGY']
    WHEEZING = request.form['WHEEZING']
    ALCOHOL_CONSUMING = request.form['ALCOHOL_CONSUMING']
    COUGHING = request.form['COUGHING']
    SHORTNESS_OF_BREATH	 = request.form['SHORTNESS_OF_BREATH']
    SWALLOWING_DIFFICULTY = request.form['SWALLOWING_DIFFICULTY']
    CHEST_PAIN = request.form['CHEST_PAIN']
    input_2=pd.DataFrame([[GENDER,AGE,SMOKING,YELLOW_FINGERS,
                           ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,
                           FATIGUE,ALLERGY,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN]],columns=['GENDER','AGE','SMOKING',
                                                                 'YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE'
                                                                 ,'FATIGUE','ALLERGY','WHEEZING','ALCOHOL CONSUMING','COUGHING'
                                                                  ,'SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN'])

    pred= model.predict(input_2)

    return render_template('result.html',value=pred)

@app.route("/breast_cancer_predict",methods=['POST'])

def breast_cancer_predict():
    radius_mean = request.form['radius_mean']
    texture_mean = request.form['texture_mean']
    perimeter_mean = request.form['perimeter_mean']
    area_mean = request.form['area_mean']
    smoothness_mean = request.form['smoothness_mean']                                                                
    compactness_mean = request.form['compactness_mean']                                                               
    concavity_mean = request.form['concavity_mean']                                                                 
    concave_points_mean = request.form['concave_points_mean']                                                                
    symmetry_mean = request.form['symmetry_mean']                                                             
    fractal_dimension_mean = request.form['fractal_dimension_mean']                                                               
    radius_se = request.form['radius_se']                                                                
    texture_se = request.form['texture_se']
    perimeter_se = request.form['perimeter_se']
    area_se = request.form['area_se']
    smoothness_se = request.form['smoothness_se']
    compactness_se = request.form['compactness_se']
    concavity_se = request.form['concavity_se']
    concave_points_se = request.form['concave_points_se']
    symmetry_se = request.form['symmetry_se']
    fractal_dimension_se= request.form['fractal_dimension_se']
    radius_worst = request.form['radius_worst']
    texture_worst = request.form['texture_worst']
    perimeter_worst = request.form['perimeter_worst']
    area_worst = request.form['area_worst']
    smoothness_worst =request.form['smoothness_worst']
    compactness_worst = request.form['compactness_worst']
    concavity_worst = request.form['concavity_worst']
    concave_points_worst= request.form['concave_points_worst']
    symmetry_worst = request.form['symmetry_worst']
    fractal_dimension_worst =request.form['fractal_dimension_worst']

    input_1 = pd.DataFrame([[radius_mean,texture_mean,perimeter_mean,area_mean,
                            smoothness_mean,compactness_mean,concavity_mean,
                            concave_points_mean,symmetry_mean,fractal_dimension_mean,
                            radius_se,texture_se,perimeter_se,area_se,smoothness_se,
                            compactness_se,concavity_se,concave_points_se,symmetry_se,
                            fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,
                            area_worst,smoothness_worst,compactness_worst,concavity_worst,
                            concave_points_worst,symmetry_worst,fractal_dimension_worst]],

                            columns=['radius_mean','texture_mean','perimeter_mean','area_mean',
                                    'smoothness_mean','compactness_mean','concavity_mean',
                                    'concave points_mean','symmetry_mean','fractal_dimension_mean',
                                    'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
                                    'compactness_se','concavity_se','concave points_se','symmetry_se',
                                    'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst',
                                    'area_worst','smoothness_worst','compactness_worst','concavity_worst',
                                    'concave points_worst','symmetry_worst','fractal_dimension_worst']
                            )
    predict_1 = mocel_1.predict(input_1)
    return render_template('Result.html',Value=predict_1)

if __name__== "__main__":
    app.run(debug=True)