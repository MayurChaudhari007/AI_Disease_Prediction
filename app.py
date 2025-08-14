from flask import Flask, request, redirect, render_template, url_for, jsonify
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
# 
# from google import genai as ge
# import math
import ast
# import openai
import google.generativeai as genai
# import requests
import json
import os
import re 
import importlib

# Load google-genai's genai
from dotenv import load_dotenv
ge = importlib.import_module("google.genai")

# Import the library

# Load environment variables from .env file
load_dotenv()


############### Api Key ###################




################ DATAsets ################

sym_desk = pd.read_csv("Dataset/symtoms_df.csv")
precautions = pd.read_csv("Dataset/precautions_df.csv")
workout = pd.read_csv("Dataset/workout_df.csv")
description = pd.read_csv("Dataset/description.csv")
medications = pd.read_csv("Dataset/medications.csv")
diets = pd.read_csv("Dataset/diets.csv")
# Extra
medications1 = pd.read_csv("Dataset/medications.csv")
medications1["Medication"] = medications1["Medication"].apply(ast.literal_eval)
diets1 = pd.read_csv("Dataset/diets.csv")
diets1["Diet"] = diets1["Diet"].apply(ast.literal_eval)


#  load model
svc = pickle.load(open("model/svc.pkl", "rb"))


# # function
def medi(dis):
    results = []
    if "Disease" in medications1.columns:
        dis_type = medications1[medications1["Disease"] == dis]
        for med_list in dis_type["Medication"]:
            for med in med_list:
                results.append(med)
    return results


def dieti(dis):
    results = []
    if "Disease" in diets1.columns:
        diet_type = diets1[diets1["Disease"] == dis]
        for diet_list in diet_type["Diet"]:
            for die in diet_list:
                results.append(die)
    return results


def helper(dis):
    desc = description[description["Disease"] == dis]["Description"]
    desc = " ".join([w for w in desc])

    pre = precautions[precautions["Disease"] == dis][
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    ]

    pre = pre.values.flatten().tolist()

    wrkout = workout[workout["disease"] == dis]["workout"]

    return desc, pre, wrkout


symptoms_dict = {
    "itching": 0,
    "skin_rash": 1,
    "nodal_skin_eruptions": 2,
    "continuous_sneezing": 3,
    "shivering": 4,
    "chills": 5,
    "joint_pain": 6,
    "stomach_pain": 7,
    "acidity": 8,
    "ulcers_on_tongue": 9,
    "muscle_wasting": 10,
    "vomiting": 11,
    "burning_micturition": 12,
    "spotting_ urination": 13,
    "fatigue": 14,
    "weight_gain": 15,
    "anxiety": 16,
    "cold_hands_and_feets": 17,
    "mood_swings": 18,
    "weight_loss": 19,
    "restlessness": 20,
    "lethargy": 21,
    "patches_in_throat": 22,
    "irregular_sugar_level": 23,
    "cough": 24,
    "high_fever": 25,
    "sunken_eyes": 26,
    "breathlessness": 27,
    "sweating": 28,
    "dehydration": 29,
    "indigestion": 30,
    "headache": 31,
    "yellowish_skin": 32,
    "dark_urine": 33,
    "nausea": 34,
    "loss_of_appetite": 35,
    "pain_behind_the_eyes": 36,
    "back_pain": 37,
    "constipation": 38,
    "abdominal_pain": 39,
    "diarrhoea": 40,
    "mild_fever": 41,
    "yellow_urine": 42,
    "yellowing_of_eyes": 43,
    "acute_liver_failure": 44,
    "fluid_overload": 45,
    "swelling_of_stomach": 46,
    "swelled_lymph_nodes": 47,
    "malaise": 48,
    "blurred_and_distorted_vision": 49,
    "phlegm": 50,
    "throat_irritation": 51,
    "redness_of_eyes": 52,
    "sinus_pressure": 53,
    "runny_nose": 54,
    "congestion": 55,
    "chest_pain": 56,
    "weakness_in_limbs": 57,
    "fast_heart_rate": 58,
    "pain_during_bowel_movements": 59,
    "pain_in_anal_region": 60,
    "bloody_stool": 61,
    "irritation_in_anus": 62,
    "neck_pain": 63,
    "dizziness": 64,
    "cramps": 65,
    "bruising": 66,
    "obesity": 67,
    "swollen_legs": 68,
    "swollen_blood_vessels": 69,
    "puffy_face_and_eyes": 70,
    "enlarged_thyroid": 71,
    "brittle_nails": 72,
    "swollen_extremeties": 73,
    "excessive_hunger": 74,
    "extra_marital_contacts": 75,
    "drying_and_tingling_lips": 76,
    "slurred_speech": 77,
    "knee_pain": 78,
    "hip_joint_pain": 79,
    "muscle_weakness": 80,
    "stiff_neck": 81,
    "swelling_joints": 82,
    "movement_stiffness": 83,
    "spinning_movements": 84,
    "loss_of_balance": 85,
    "unsteadiness": 86,
    "weakness_of_one_body_side": 87,
    "loss_of_smell": 88,
    "bladder_discomfort": 89,
    "foul_smell_of urine": 90,
    "continuous_feel_of_urine": 91,
    "passage_of_gases": 92,
    "internal_itching": 93,
    "toxic_look_(typhos)": 94,
    "depression": 95,
    "irritability": 96,
    "muscle_pain": 97,
    "altered_sensorium": 98,
    "red_spots_over_body": 99,
    "belly_pain": 100,
    "abnormal_menstruation": 101,
    "dischromic _patches": 102,
    "watering_from_eyes": 103,
    "increased_appetite": 104,
    "polyuria": 105,
    "family_history": 106,
    "mucoid_sputum": 107,
    "rusty_sputum": 108,
    "lack_of_concentration": 109,
    "visual_disturbances": 110,
    "receiving_blood_transfusion": 111,
    "receiving_unsterile_injections": 112,
    "coma": 113,
    "stomach_bleeding": 114,
    "distention_of_abdomen": 115,
    "history_of_alcohol_consumption": 116,
    "fluid_overload.1": 117,
    "blood_in_sputum": 118,
    "prominent_veins_on_calf": 119,
    "palpitations": 120,
    "painful_walking": 121,
    "pus_filled_pimples": 122,
    "blackheads": 123,
    "scurring": 124,
    "skin_peeling": 125,
    "silver_like_dusting": 126,
    "small_dents_in_nails": 127,
    "inflammatory_nails": 128,
    "blister": 129,
    "red_sore_around_nose": 130,
    "yellow_crust_ooze": 131,
}
diseases_list = {
    15: "Fungal infection",
    4: "Allergy",
    16: "GERD",
    9: "Chronic cholestasis",
    14: "Drug Reaction",
    33: "Peptic ulcer diseae",
    1: "AIDS",
    12: "Diabetes ",
    17: "Gastroenteritis",
    6: "Bronchial Asthma",
    23: "Hypertension ",
    30: "Migraine",
    7: "Cervical spondylosis",
    32: "Paralysis (brain hemorrhage)",
    28: "Jaundice",
    29: "Malaria",
    8: "Chicken pox",
    11: "Dengue",
    37: "Typhoid",
    40: "hepatitis A",
    19: "Hepatitis B",
    20: "Hepatitis C",
    21: "Hepatitis D",
    22: "Hepatitis E",
    3: "Alcoholic hepatitis",
    36: "Tuberculosis",
    10: "Common Cold",
    34: "Pneumonia",
    13: "Dimorphic hemmorhoids(piles)",
    18: "Heart attack",
    39: "Varicose veins",
    26: "Hypothyroidism",
    24: "Hyperthyroidism",
    25: "Hypoglycemia",
    31: "Osteoarthristis",
    5: "Arthritis",
    0: "(vertigo) Paroymsal  Positional Vertigo",
    2: "Acne",
    38: "Urinary tract infection",
    35: "Psoriasis",
    27: "Impetigo",
}


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


# #############################


# ############ AI Prediction Functions #########################


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Configure the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-lite",
    system_instruction="""
    You are a professional AI doctor.
The user will give you their name, age, gender, and symptoms.

You must respond in the following JSON format only:
{
  "patient_name": "<name>",
  "predicted_disease": "<disease name>",
  "medical_advice": ["advice 1", "advice 2", "advice 3"],
  "additional_info": ["extra info 1", "extra info 2"],
  "doctor_alerts": ["alert 1", "alert 2"]
}

Rules:
- Do not include any extra text outside the JSON dont add json variable while creating json file.
- Keep each advice short and clear.
- Use layman-friendly language.
- If unsure, say “Possible causes include...” in the disease field.
    
    """,
)





# ############ Chat Bot Functions #########################

client = ge.Client(api_key=os.getenv("GEMINI_API_KEY"))


chat = client.chats.create(
    model="gemini-2.0-flash-lite",
    history=[
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "You are a friendly AI doctor. "
                        "Ask clear, short questions if needed. "
                        "Give simple and easy-to-understand medical advice. "
                        "used less line to give answer like 2 or 3 line only"
                        "Also give medical or health advice if neccessary "
                        "Be polite, helpful, and concise."
                    )
                }
            ]
        }
    ]
)

########################################################

app = Flask(__name__)

# #############################


########################################################
#####################    Home      #######################################


@app.route("/")
def home():
    return render_template("home.html")


########################################################
#####################    predict      #######################################


@app.route("/predict", methods=["GET", "POST"])
def form():

    if request.method == "POST":
        name = request.form.get("patient_name")
        age = request.form.get("patient_age")
        gender = request.form.get("patient_gender")
        symptoms = request.form.getlist("symptoms")
        if not symptoms:
            message = "Please select at least one symptom."
            return render_template(
                "predict.html",
                message=message,
                symptoms_list=list(symptoms_dict.keys()),
            )

        # Predict disease
        predicted_disease = get_predicted_value(symptoms)
        desc, pre, wrkout = helper(predicted_disease)
        medi2 = medi(predicted_disease)
        diet2 = dieti(predicted_disease)
        report_time = datetime.now().strftime("%d-%m-%Y %H:%M")
        return render_template(
            "predict.html",
            name=name,
            age=age,
            gender=gender,
            symptoms=symptoms,
            predicted_disease=predicted_disease,
            report_time=report_time,
            dis_desc=desc,
            dis_prec=pre,
            dis_medi=medi2,
            dis_diet=diet2,
            dis_wrk=wrkout,
            symptoms_list=list(symptoms_dict.keys()),
        )
    return render_template("predict.html", symptoms_list=list(symptoms_dict.keys()))


########################################################################################
######################     AI   Prediction       #######################################




@app.route('/ai_predict', methods=['GET', 'POST'])
def ai_predict():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        symptoms = request.form.get('symptoms')

        user_prompt = f"Name: {name}, Age: {age}, Gender: {gender}, Symptoms: {symptoms}"

        # Get AI response
        response = model.generate_content(user_prompt)

        
        
        prediction = response.text
        ######################
        cleaned_text = re.sub(r"^```json|```$", "", prediction.strip())
        cleaned_text = cleaned_text.strip("`").strip()
        
        # 2. Try to parse into Python dict
        try:
            prediction_data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            prediction_data = {"raw_text": cleaned_text}

        result = prediction_data
        report_time = datetime.now().strftime("%d-%m-%Y %H:%M")
        ##################################
        # return render_template("ai_predict.html", prediction=prediction)
        return render_template("ai_predict.html", result=result,report_time=report_time)

    return render_template("ai_predict.html", prediction=None)


##########################################################################################
######################     Chat Bot       #######################################

@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("message", "")
    if not user_message.strip():
        return jsonify({"reply": "Please type something."})

    res = chat.send_message(user_message)
    bot_reply = res.text.strip()
    return jsonify({"reply": bot_reply})








##########################################################################################
@app.route("/tutor")
def tutor():
    return render_template("tutor.html")


##########################################################################################
@app.route("/about")
def about():
    return render_template("about.html")


##########################################################################################
@app.route("/contact")
def contact():
    return render_template("contact.html")


##########################################################################################
if __name__ == "__main__":
    app.run(debug=True)
