import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import subprocess




st.set_page_config(
    page_title="Disease Finder Tool",
    page_icon="🦠",
)
st.sidebar.success("Select a page above.")

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

z = st.selectbox(
   "What is Your First Symptom?",
   ("Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing", "Shivering", "Chills", "Joint Pain", "Stomach Pain", "Acidity", "Ulcers On Tongue", "Muscle Wasting", "Vomiting", "Burning Micturition", "Spotting Urination", "Fatigue", "Weight Gain", "Anxiety", "Cold Hands And Feets", "Mood Swings", "Weight Loss", "Restlessness", "Lethargy", "Patches In Throat", "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes", "Breathlessness", "Sweating", "Dehydration", "Indigestion", "Headache", "Yellowish Skin", "Dark Urine", "Nausea", "Loss Of Appetite", "Pain Behind The Eyes", "Back Pain", "Constipation", "Abdominal Pain", "Diarhoea", "Mild Fever", "Yellow Urine", "Yellowing Of Eyes", "Acute Liver Failure", "Fluid Overload", "Swelling Of Stomach", "Swelled Lymph Nodes", "Malaise", "Blurred And Distorted Vision", "Phlegm", "Throat Irritation", "Redness Of Eyes", "Sinus Pressure", "Runny Nose", "Congestion", "Chest Pain", "Weakness In Limbs", "Fast Heart Rate", "Pain During Bowel Movements", "Pain In Anal Region", "Bloody Stool", "Irritaion In Anus", "Neck Pain", "Dizziness", "Cramps", "Bruising", "Obesity", "Swollen Legs", "Swollen Blood Vessels", "Puffy Face And Eyes", "Enlarged Thyroid", "Brittle Nails", "Swollen Extremeties", "Excessive Hunger", "Extra Marital Contacts", "Drying And Tingling Lips", "Slurred Speech", "Knee Pain", "Hip Joint Pain", "Muscle Weakness", "Stiff Neck", "Swelling Joints", "Movement Stiffness", "Spinning Movements", "Loss Of Balance", "Unsteadiness", "Weakness Of One Body Side", "Loss Of Smell", "Bladder Discomfort", "Foul Smell Of Urine", "Continuous Feel Of Urine", "Passage Of Gases", "Internal Itching", "Toxic Look (Typhos)", "Depression", "Irritablity", "Muscle Pain", "Altered Sensorium", "Red Spots Over Body", "Belly Pain", "Abnormal Mensuration", "Dischromic Patches", "Watering From Eyes", "Increased Appetite", "Polyuria", "Family History", "Mucoid Sputum", "Rusty Sputum", "Lack Of Concentration", "Visual Disturbances", "Receiving Blood Transfusion", "Receiving Unsterile Injections", "Coma", "Stomach Bleeding", "Distention Of Abdomen", "History of Alcohol Consumption", "Fluid Overload", "Blood In Sputum", "Prominent Veins On Calf", "Palpitations", "Painful Walking", "Pus Filled Pimples", "Blackheads", "Scurring", "Skin Peeling", "Silver Like Dusting", "Small Dents In Nails", "Inflammatory Nails", "Blister", "Red Sore Around Nose", "Yellow Crust Ooze", "None Of These"),
   index=None,
   placeholder="Select Your Symptom",
)

st.write('You selected:', z)
y = st.selectbox(
   "What is Your Second Symptom?",
   ("Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing", "Shivering", "Chills", "Joint Pain", "Stomach Pain", "Acidity", "Ulcers On Tongue", "Muscle Wasting", "Vomiting", "Burning Micturition", "Spotting Urination", "Fatigue", "Weight Gain", "Anxiety", "Cold Hands And Feets", "Mood Swings", "Weight Loss", "Restlessness", "Lethargy", "Patches In Throat", "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes", "Breathlessness", "Sweating", "Dehydration", "Indigestion", "Headache", "Yellowish Skin", "Dark Urine", "Nausea", "Loss Of Appetite", "Pain Behind The Eyes", "Back Pain", "Constipation", "Abdominal Pain", "Diarhoea", "Mild Fever", "Yellow Urine", "Yellowing Of Eyes", "Acute Liver Failure", "Fluid Overload", "Swelling Of Stomach", "Swelled Lymph Nodes", "Malaise", "Blurred And Distorted Vision", "Phlegm", "Throat Irritation", "Redness Of Eyes", "Sinus Pressure", "Runny Nose", "Congestion", "Chest Pain", "Weakness In Limbs", "Fast Heart Rate", "Pain During Bowel Movements", "Pain In Anal Region", "Bloody Stool", "Irritaion In Anus", "Neck Pain", "Dizziness", "Cramps", "Bruising", "Obesity", "Swollen Legs", "Swollen Blood Vessels", "Puffy Face And Eyes", "Enlarged Thyroid", "Brittle Nails", "Swollen Extremeties", "Excessive Hunger", "Extra Marital Contacts", "Drying And Tingling Lips", "Slurred Speech", "Knee Pain", "Hip Joint Pain", "Muscle Weakness", "Stiff Neck", "Swelling Joints", "Movement Stiffness", "Spinning Movements", "Loss Of Balance", "Unsteadiness", "Weakness Of One Body Side", "Loss Of Smell", "Bladder Discomfort", "Foul Smell Of Urine", "Continuous Feel Of Urine", "Passage Of Gases", "Internal Itching", "Toxic Look (Typhos)", "Depression", "Irritablity", "Muscle Pain", "Altered Sensorium", "Red Spots Over Body", "Belly Pain", "Abnormal Mensuration", "Dischromic Patches", "Watering From Eyes", "Increased Appetite", "Polyuria", "Family History", "Mucoid Sputum", "Rusty Sputum", "Lack Of Concentration", "Visual Disturbances", "Receiving Blood Transfusion", "Receiving Unsterile Injections", "Coma", "Stomach Bleeding", "Distention Of Abdomen", "History of Alcohol Consumption", "Fluid Overload", "Blood In Sputum", "Prominent Veins On Calf", "Palpitations", "Painful Walking", "Pus Filled Pimples", "Blackheads", "Scurring", "Skin Peeling", "Silver Like Dusting", "Small Dents In Nails", "Inflammatory Nails", "Blister", "Red Sore Around Nose", "Yellow Crust Ooze", "None Of These"),
   index=None,
   placeholder="Select Your Symptom",
)

st.write('You selected:', y)
x = st.selectbox(
   "What is Your Third Symptom?",
   ("Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing", "Shivering", "Chills", "Joint Pain", "Stomach Pain", "Acidity", "Ulcers On Tongue", "Muscle Wasting", "Vomiting", "Burning Micturition", "Spotting Urination", "Fatigue", "Weight Gain", "Anxiety", "Cold Hands And Feets", "Mood Swings", "Weight Loss", "Restlessness", "Lethargy", "Patches In Throat", "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes", "Breathlessness", "Sweating", "Dehydration", "Indigestion", "Headache", "Yellowish Skin", "Dark Urine", "Nausea", "Loss Of Appetite", "Pain Behind The Eyes", "Back Pain", "Constipation", "Abdominal Pain", "Diarhoea", "Mild Fever", "Yellow Urine", "Yellowing Of Eyes", "Acute Liver Failure", "Fluid Overload", "Swelling Of Stomach", "Swelled Lymph Nodes", "Malaise", "Blurred And Distorted Vision", "Phlegm", "Throat Irritation", "Redness Of Eyes", "Sinus Pressure", "Runny Nose", "Congestion", "Chest Pain", "Weakness In Limbs", "Fast Heart Rate", "Pain During Bowel Movements", "Pain In Anal Region", "Bloody Stool", "Irritaion In Anus", "Neck Pain", "Dizziness", "Cramps", "Bruising", "Obesity", "Swollen Legs", "Swollen Blood Vessels", "Puffy Face And Eyes", "Enlarged Thyroid", "Brittle Nails", "Swollen Extremeties", "Excessive Hunger", "Extra Marital Contacts", "Drying And Tingling Lips", "Slurred Speech", "Knee Pain", "Hip Joint Pain", "Muscle Weakness", "Stiff Neck", "Swelling Joints", "Movement Stiffness", "Spinning Movements", "Loss Of Balance", "Unsteadiness", "Weakness Of One Body Side", "Loss Of Smell", "Bladder Discomfort", "Foul Smell Of Urine", "Continuous Feel Of Urine", "Passage Of Gases", "Internal Itching", "Toxic Look (Typhos)", "Depression", "Irritablity", "Muscle Pain", "Altered Sensorium", "Red Spots Over Body", "Belly Pain", "Abnormal Mensuration", "Dischromic Patches", "Watering From Eyes", "Increased Appetite", "Polyuria", "Family History", "Mucoid Sputum", "Rusty Sputum", "Lack Of Concentration", "Visual Disturbances", "Receiving Blood Transfusion", "Receiving Unsterile Injections", "Coma", "Stomach Bleeding", "Distention Of Abdomen", "History of Alcohol Consumption", "Fluid Overload", "Blood In Sputum", "Prominent Veins On Calf", "Palpitations", "Painful Walking", "Pus Filled Pimples", "Blackheads", "Scurring", "Skin Peeling", "Silver Like Dusting", "Small Dents In Nails", "Inflammatory Nails", "Blister", "Red Sore Around Nose", "Yellow Crust Ooze", "None Of These"),
   index=None,
   placeholder="Select Your Symptom",
)

st.write('You selected:', x)
w = st.selectbox(
   "What is Your Fourth Symptom??",
   ("Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing", "Shivering", "Chills", "Joint Pain", "Stomach Pain", "Acidity", "Ulcers On Tongue", "Muscle Wasting", "Vomiting", "Burning Micturition", "Spotting Urination", "Fatigue", "Weight Gain", "Anxiety", "Cold Hands And Feets", "Mood Swings", "Weight Loss", "Restlessness", "Lethargy", "Patches In Throat", "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes", "Breathlessness", "Sweating", "Dehydration", "Indigestion", "Headache", "Yellowish Skin", "Dark Urine", "Nausea", "Loss Of Appetite", "Pain Behind The Eyes", "Back Pain", "Constipation", "Abdominal Pain", "Diarhoea", "Mild Fever", "Yellow Urine", "Yellowing Of Eyes", "Acute Liver Failure", "Fluid Overload", "Swelling Of Stomach", "Swelled Lymph Nodes", "Malaise", "Blurred And Distorted Vision", "Phlegm", "Throat Irritation", "Redness Of Eyes", "Sinus Pressure", "Runny Nose", "Congestion", "Chest Pain", "Weakness In Limbs", "Fast Heart Rate", "Pain During Bowel Movements", "Pain In Anal Region", "Bloody Stool", "Irritaion In Anus", "Neck Pain", "Dizziness", "Cramps", "Bruising", "Obesity", "Swollen Legs", "Swollen Blood Vessels", "Puffy Face And Eyes", "Enlarged Thyroid", "Brittle Nails", "Swollen Extremeties", "Excessive Hunger", "Extra Marital Contacts", "Drying And Tingling Lips", "Slurred Speech", "Knee Pain", "Hip Joint Pain", "Muscle Weakness", "Stiff Neck", "Swelling Joints", "Movement Stiffness", "Spinning Movements", "Loss Of Balance", "Unsteadiness", "Weakness Of One Body Side", "Loss Of Smell", "Bladder Discomfort", "Foul Smell Of Urine", "Continuous Feel Of Urine", "Passage Of Gases", "Internal Itching", "Toxic Look (Typhos)", "Depression", "Irritablity", "Muscle Pain", "Altered Sensorium", "Red Spots Over Body", "Belly Pain", "Abnormal Mensuration", "Dischromic Patches", "Watering From Eyes", "Increased Appetite", "Polyuria", "Family History", "Mucoid Sputum", "Rusty Sputum", "Lack Of Concentration", "Visual Disturbances", "Receiving Blood Transfusion", "Receiving Unsterile Injections", "Coma", "Stomach Bleeding", "Distention Of Abdomen", "History of Alcohol Consumption", "Fluid Overload", "Blood In Sputum", "Prominent Veins On Calf", "Palpitations", "Painful Walking", "Pus Filled Pimples", "Blackheads", "Scurring", "Skin Peeling", "Silver Like Dusting", "Small Dents In Nails", "Inflammatory Nails", "Blister", "Red Sore Around Nose", "Yellow Crust Ooze", "None Of These"),
   index=None,
   placeholder="Select Your Symptom",
)

st.write('You selected:', w)
v = st.selectbox(
   "What is Your Fifth Symptom?",
   ("Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing", "Shivering", "Chills", "Joint Pain", "Stomach Pain", "Acidity", "Ulcers On Tongue", "Muscle Wasting", "Vomiting", "Burning Micturition", "Spotting Urination", "Fatigue", "Weight Gain", "Anxiety", "Cold Hands And Feets", "Mood Swings", "Weight Loss", "Restlessness", "Lethargy", "Patches In Throat", "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes", "Breathlessness", "Sweating", "Dehydration", "Indigestion", "Headache", "Yellowish Skin", "Dark Urine", "Nausea", "Loss Of Appetite", "Pain Behind The Eyes", "Back Pain", "Constipation", "Abdominal Pain", "Diarhoea", "Mild Fever", "Yellow Urine", "Yellowing Of Eyes", "Acute Liver Failure", "Fluid Overload", "Swelling Of Stomach", "Swelled Lymph Nodes", "Malaise", "Blurred And Distorted Vision", "Phlegm", "Throat Irritation", "Redness Of Eyes", "Sinus Pressure", "Runny Nose", "Congestion", "Chest Pain", "Weakness In Limbs", "Fast Heart Rate", "Pain During Bowel Movements", "Pain In Anal Region", "Bloody Stool", "Irritaion In Anus", "Neck Pain", "Dizziness", "Cramps", "Bruising", "Obesity", "Swollen Legs", "Swollen Blood Vessels", "Puffy Face And Eyes", "Enlarged Thyroid", "Brittle Nails", "Swollen Extremeties", "Excessive Hunger", "Extra Marital Contacts", "Drying And Tingling Lips", "Slurred Speech", "Knee Pain", "Hip Joint Pain", "Muscle Weakness", "Stiff Neck", "Swelling Joints", "Movement Stiffness", "Spinning Movements", "Loss Of Balance", "Unsteadiness", "Weakness Of One Body Side", "Loss Of Smell", "Bladder Discomfort", "Foul Smell Of Urine", "Continuous Feel Of Urine", "Passage Of Gases", "Internal Itching", "Toxic Look (Typhos)", "Depression", "Irritablity", "Muscle Pain", "Altered Sensorium", "Red Spots Over Body", "Belly Pain", "Abnormal Mensuration", "Dischromic Patches", "Watering From Eyes", "Increased Appetite", "Polyuria", "Family History", "Mucoid Sputum", "Rusty Sputum", "Lack Of Concentration", "Visual Disturbances", "Receiving Blood Transfusion", "Receiving Unsterile Injections", "Coma", "Stomach Bleeding", "Distention Of Abdomen", "History of Alcohol Consumption", "Fluid Overload", "Blood In Sputum", "Prominent Veins On Calf", "Palpitations", "Painful Walking", "Pus Filled Pimples", "Blackheads", "Scurring", "Skin Peeling", "Silver Like Dusting", "Small Dents In Nails", "Inflammatory Nails", "Blister", "Red Sore Around Nose", "Yellow Crust Ooze", "None Of These"),
   index=None,
   placeholder="Select Your Symptom",
)

st.write('You selected:', v)
u = st.selectbox(
   "What is Your Sixth Symptom?",
   ("Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing", "Shivering", "Chills", "Joint Pain", "Stomach Pain", "Acidity", "Ulcers On Tongue", "Muscle Wasting", "Vomiting", "Burning Micturition", "Spotting Urination", "Fatigue", "Weight Gain", "Anxiety", "Cold Hands And Feets", "Mood Swings", "Weight Loss", "Restlessness", "Lethargy", "Patches In Throat", "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes", "Breathlessness", "Sweating", "Dehydration", "Indigestion", "Headache", "Yellowish Skin", "Dark Urine", "Nausea", "Loss Of Appetite", "Pain Behind The Eyes", "Back Pain", "Constipation", "Abdominal Pain", "Diarhoea", "Mild Fever", "Yellow Urine", "Yellowing Of Eyes", "Acute Liver Failure", "Fluid Overload", "Swelling Of Stomach", "Swelled Lymph Nodes", "Malaise", "Blurred And Distorted Vision", "Phlegm", "Throat Irritation", "Redness Of Eyes", "Sinus Pressure", "Runny Nose", "Congestion", "Chest Pain", "Weakness In Limbs", "Fast Heart Rate", "Pain During Bowel Movements", "Pain In Anal Region", "Bloody Stool", "Irritaion In Anus", "Neck Pain", "Dizziness", "Cramps", "Bruising", "Obesity", "Swollen Legs", "Swollen Blood Vessels", "Puffy Face And Eyes", "Enlarged Thyroid", "Brittle Nails", "Swollen Extremeties", "Excessive Hunger", "Extra Marital Contacts", "Drying And Tingling Lips", "Slurred Speech", "Knee Pain", "Hip Joint Pain", "Muscle Weakness", "Stiff Neck", "Swelling Joints", "Movement Stiffness", "Spinning Movements", "Loss Of Balance", "Unsteadiness", "Weakness Of One Body Side", "Loss Of Smell", "Bladder Discomfort", "Foul Smell Of Urine", "Continuous Feel Of Urine", "Passage Of Gases", "Internal Itching", "Toxic Look (Typhos)", "Depression", "Irritablity", "Muscle Pain", "Altered Sensorium", "Red Spots Over Body", "Belly Pain", "Abnormal Mensuration", "Dischromic Patches", "Watering From Eyes", "Increased Appetite", "Polyuria", "Family History", "Mucoid Sputum", "Rusty Sputum", "Lack Of Concentration", "Visual Disturbances", "Receiving Blood Transfusion", "Receiving Unsterile Injections", "Coma", "Stomach Bleeding", "Distention Of Abdomen", "History of Alcohol Consumption", "Fluid Overload", "Blood In Sputum", "Prominent Veins On Calf", "Palpitations", "Painful Walking", "Pus Filled Pimples", "Blackheads", "Scurring", "Skin Peeling", "Silver Like Dusting", "Small Dents In Nails", "Inflammatory Nails", "Blister", "Red Sore Around Nose", "Yellow Crust Ooze", "None Of These"),
    index=None,
    placeholder="Select Your Symptom",
)

st.write('You selected:', u)
if z!=None and y!=None and x!=None and w!=None and v!=None and u!=None:
    a = str(z) 
    b = str(y)
    c = str(x)
    d = str(w)
    e = str(v)
    f = str(u)

    print(a, b, c, d, e, f)

    st.write("Analyzing Given Information And Preparing Results For You...")






    warnings.filterwarnings("ignore")


    #%matplotlib inline



        
    DATA_PATH = "C:\\SymptoDetect\\Training.csv"
    data = pd.read_csv(DATA_PATH).dropna(axis = 1)

    disease_counts = data["prognosis"].value_counts()
    temp_df = pd.DataFrame({
        "Disease": disease_counts.index,
        "Counts": disease_counts.values
    })
    


    encoder = LabelEncoder()
    data["prognosis"] = encoder.fit_transform(data["prognosis"])

    X = data.iloc[:,:-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test =train_test_split(
    X, y, test_size = 0.2, random_state = 24)

    def cv_scoring(estimator, X, y):
        return accuracy_score(y, estimator.predict(X))

    models = {
        "SVC":SVC(),
        "Gaussian NB":GaussianNB(),
        "Random Forest":RandomForestClassifier(random_state=18)
        }

    for model_name in models:
        model = models[model_name]
        scores = cross_val_score(model, X, y, cv = 10, 
                                n_jobs = -1, 
                                scoring = cv_scoring)
        print("=="*30)
        print(model_name)
        print(f"Scores: {scores}")
        print(f"Mean Score: {np.mean(scores)}")

    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    preds = svm_model.predict(X_test)
    
    print(f"Accuracy on train data by SVM Classifier\
    : {accuracy_score(y_train, svm_model.predict(X_train))*100}")
    
    print(f"Accuracy on test data by SVM Classifier\
    : {accuracy_score(y_test, preds)*100}")
    cf_matrix = confusion_matrix(y_test, preds)

    
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    preds = nb_model.predict(X_test)
    print(f"Accuracy on train data by Naive Bayes Classifier\
    : {accuracy_score(y_train, nb_model.predict(X_train))*100}")
    
    print(f"Accuracy on test data by Naive Bayes Classifier\
    : {accuracy_score(y_test, preds)*100}")
    cf_matrix = confusion_matrix(y_test, preds)

    
    rf_model = RandomForestClassifier(random_state=18)
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict(X_test)
    print(f"Accuracy on train data by Random Forest Classifier\
    : {accuracy_score(y_train, rf_model.predict(X_train))*100}")
    
    print(f"Accuracy on test data by Random Forest Classifier\
    : {accuracy_score(y_test, preds)*100}")
    
    cf_matrix = confusion_matrix(y_test, preds)


    final_svm_model = SVC()
    final_nb_model = GaussianNB()
    final_rf_model = RandomForestClassifier(random_state=18)
    final_svm_model.fit(X, y)
    final_nb_model.fit(X, y)
    final_rf_model.fit(X, y)
    

    test_data = pd.read_csv("C:\\SymptoDetect\\Testing.csv").dropna(axis=1)
    
    test_X = test_data.iloc[:, :-1]
    test_Y = encoder.transform(test_data.iloc[:, -1])
    

    svm_preds = final_svm_model.predict(test_X)
    nb_preds = final_nb_model.predict(test_X)
    rf_preds = final_rf_model.predict(test_X)
    

    final_preds = [mode([i,j,k])[0] for i,j,
                k in zip(svm_preds, nb_preds, rf_preds)]
    
    print(f"Accuracy on Test dataset by the combined model\
    : {accuracy_score(test_Y, final_preds)*100}")
    
    cf_matrix = confusion_matrix(test_Y, final_preds)


    symptoms = X.columns.values

    symptom_index = {}
    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index
    
    data_dict = {
        "symptom_index":symptom_index,
        "predictions_classes":encoder.classes_
    }






    def predictDisease(symptoms):
        symptoms = symptoms.split(",")
        

        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1

        input_data = np.array(input_data).reshape(1,-1)
        
        rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
        nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
        svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

        final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
        predictions = {
            "Might Get A Checkup For": rf_prediction,
            "and": nb_prediction,
            "and ": svm_prediction,
            #"Get A Checkup For":final_prediction
        }
        return predictions

    g=a+","+b+","+c+","+d+","+e
    

    if a=="None Of These" and b!="None Of These" and c!="None Of These" and d!="None Of These" and e!="None Of These" and f!="None Of These":
        h=b+","+c+","+d+","+e+","+f
        callfun = predictDisease(h)
        print(predictDisease(h))
        st.write(predictDisease(h))
    if b=="None Of These" and a!="None Of These" and c!="None Of These" and d!="None Of These" and e!="None Of These" and f!="None Of These":
        i=a+","+c+","+d+","+e+","+f
        callfun = predictDisease(i)
        print(predictDisease(i))
        st.write(predictDisease(i))
    if c=="None Of These" and b!="None Of These" and a!="None Of These" and d!="None Of These" and e!="None Of These" and f!="None Of These":
        j=a+","+b+","+d+","+e+","+f
        callfun = predictDisease(j)
        print(predictDisease(j))
        st.write(predictDisease(j))
    if d=="None Of These" and b!="None Of These" and c!="None Of These" and a!="None Of These" and e!="None Of These" and f!="None Of These":
        k=a+","+b+","+c+","+e+","+f
        callfun = predictDisease(k)
        print(predictDisease(k))
        st.write(predictDisease(k))
    if e=="None Of These" and b!="None Of These" and c!="None Of These" and d!="None Of These" and a!="None Of These" and f!="None Of These":
        l=a+","+b+","+c+","+d+","+f
        callfun = predictDisease(l)
        print(predictDisease(l))
        st.write(predictDisease(l))
    if f=="None Of These" and b!="None Of These" and c!="None Of These" and d!="None Of These" and e!="None Of These" and a!="None Of These":
        m=a+","+b+","+c+","+d+","+e
        callfun = predictDisease(m)
        print(predictDisease(m))
        st.write(predictDisease(m))
    if a=="None Of These" and b=="None Of These":
        n=c+","+d+","+e+","+f
        callfun = predictDisease(n)
        print(predictDisease(n))
        st.write(predictDisease(n))
    if a=="None Of These" and c=="None Of These":
        o=b+","+d+","+e+","+f
        callfun = predictDisease(o)
        print(predictDisease(o))
        st.write(predictDisease(o))
    if a=="None Of These" and d=="None Of These":
        p=b+","+c+","+e+","+f
        callfun = predictDisease(p)
        print(predictDisease(p))
        st.write(predictDisease(p))
    if a=="None Of These" and e=="None Of These":
        q=b+","+c+","+d+","+f
        callfun = predictDisease(q)
        print(predictDisease(q))
        st.write(predictDisease(q))
    if a=="None Of These" and f=="None Of These":
        r=b+","+c+","+d+","+e
        callfun = predictDisease(r)
        print(predictDisease(r))
        st.write(predictDisease(r))
    if b=="None Of These" and c=="None Of These":
        t=a+","+d+","+e+","+f
        callfun = predictDisease(t)
        print(predictDisease(t))
        st.write(predictDisease(t))
    if b=="None Of These" and d=="None Of These":
        u=c+","+a+","+e+","+f
        callfun = predictDisease(u)
        print(predictDisease(u))
        st.write(predictDisease(u))
    if b=="None Of These" and e=="None Of These":
        v=c+","+d+","+a+","+f
        callfun = predictDisease(v)
        print(predictDisease(v))
        st.write(predictDisease(v))
    if b=="None Of These" and f=="None Of These":
        w=c+","+d+","+e+","+a
        callfun = predictDisease(w)
        print(predictDisease(w))
        st.write(predictDisease(w))
    if c=="None Of These" and d=="None Of These":
        x=b+","+a+","+e+","+f
        callfun = predictDisease(x)
        print(predictDisease(x))
        st.write(predictDisease(x))
    if c=="None Of These" and e=="None Of These":
        y=b+","+d+","+a+","+f
        callfun = predictDisease(y)
        print(predictDisease(y))
        st.write(predictDisease(y))
    if c=="None Of These" and f=="None Of These":
        z=b+","+d+","+e+","+a
        callfun = predictDisease(z)
        print(predictDisease(z))
        st.write(predictDisease(z))
    if d=="None Of These" and e=="None Of These":
        aa=b+","+d+","+a+","+c
        callfun = predictDisease(aa)
        print(predictDisease(aa))
        st.write(predictDisease(aa))
    if d=="None Of These" and f=="None Of These":
        ab=b+","+c+","+e+","+a
        callfun = predictDisease(ab)
        print(predictDisease(ab))
        st.write(predictDisease(ab))
    if e=="None Of These" and f=="None Of These":
        ac=b+","+c+","+d+","+a
        callfun = predictDisease(ac)
        print(predictDisease(ac))
        st.write(predictDisease(ac))

        





    if a!="None Of These" and b!="None Of These" and c!="None Of These" and d!="None Of These" and e!="None Of These" and f!="None Of These":
        print(predictDisease(g))
        st.write(predictDisease(g))

    st.write("Fetched Results For You!")    
