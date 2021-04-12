import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import numpy as np

symptoms_dict = {'abdominal_pain': 39,'abnormal_menstruation': 101,'acidity': 8,'acute_liver_failure': 44,'altered_sensorium': 98,
'anxiety': 16,'back_pain': 37,'belly_pain': 100,'blackheads': 123,'bladder_discomfort': 89,'blister': 129,'blood_in_sputum': 118,
 'bloody_stool': 61,'blurred_and_distorted_vision': 49, 'breathlessness': 27,'brittle_nails': 72,'bruising': 66,'burning_micturition': 12,
 'chest_pain': 56,'chills': 5,'cold_hands_and_feets': 17,'coma': 113,'congestion': 55,'constipation': 38,'continuous_feel_of_urine': 91,
 'continuous_sneezing': 3,'cough': 24,'cramps': 65,'dark_urine': 33,'dehydration': 29,'depression': 95,'diarrhoea': 40, 'dischromic _patches': 102,
 'distention_of_abdomen': 115, 'dizziness': 64,'drying_and_tingling_lips': 76,'enlarged_thyroid': 71,'excessive_hunger': 74,
 'extra_marital_contacts': 75,'family_history': 106,'fast_heart_rate': 58,'fatigue': 14,'fluid_overload': 45,'fluid_overload.1': 117, 'foul_smell_of urine': 90,
 'headache': 31,'high_fever': 25,'hip_joint_pain': 79, 'history_of_alcohol_consumption': 116,'increased_appetite': 104,'indigestion': 30,
 'inflammatory_nails': 128,'internal_itching': 93,'irregular_sugar_level': 23, 'irritability': 96,'irritation_in_anus': 62,
 'itching': 0,'joint_pain': 6,'knee_pain': 78,'lack_of_concentration': 109,'lethargy': 21,'loss_of_appetite': 35,'loss_of_balance': 85,
 'loss_of_smell': 88,'malaise': 48,'mild_fever': 41,'mood_swings': 18,'movement_stiffness': 83,'mucoid_sputum': 107,'muscle_pain': 97,
 'muscle_wasting': 10, 'muscle_weakness': 80, 'nausea': 34, 'neck_pain': 63,'nodal_skin_eruptions': 2,'obesity': 67,'pain_behind_the_eyes': 36,
 'pain_during_bowel_movements': 59,'pain_in_anal_region': 60,'painful_walking': 121,'palpitations': 120,'passage_of_gases': 92,
 'patches_in_throat': 22,'phlegm': 50,'polyuria': 105,'prominent_veins_on_calf': 119,'puffy_face_and_eyes': 70,'pus_filled_pimples': 122,
 'receiving_blood_transfusion': 111,'receiving_unsterile_injections': 112,'red_sore_around_nose': 130,'red_spots_over_body': 99,
 'redness_of_eyes': 52,'restlessness': 20,'runny_nose': 54,'rusty_sputum': 108,'scurring': 124,'shivering': 4,'silver_like_dusting': 126,
 'sinus_pressure': 53,'skin_peeling': 125,'skin_rash': 1,'slurred_speech': 77,'small_dents_in_nails': 127,'spinning_movements': 84,
 'spotting_ urination': 13,'stiff_neck': 81,'stomach_bleeding': 114,'stomach_pain': 7,'sunken_eyes': 26,'sweating': 28,'swelled_lymph_nodes': 47,
 'swelling_joints': 82,'swelling_of_stomach': 46,'swollen_blood_vessels': 69,'swollen_extremeties': 73,'swollen_legs': 68,
 'throat_irritation': 51,'toxic_look_(typhos)': 94,'ulcers_on_tongue': 9,'unsteadiness': 86,'visual_disturbances': 110,'vomiting': 11,'watering_from_eyes': 103,'weakness_in_limbs': 57,'weakness_of_one_body_side': 87,
 'weight_gain': 15,'weight_loss': 19,'yellow_crust_ooze': 131,'yellow_urine': 42,'yellowing_of_eyes': 43,
'yellowish_skin': 32
 }

precation_dict = {
'Drug Reaction':['Stop irritation','Stop taking drug','Consult nearest hospital'],
'Malaria': ['Keep mosquitos out', 'Avoid oily food', 'Consult nearest hospital'],
'Allergy': ['Apply calamine', 'Cover area with bandage', 'Use ice to compress itching'],
'Hypothyroidism': ['Reduce stress', 'Exercise', 'Eat Health', 'Get proper sleep'],
'Psoriasis': ['Wash hands with warm soapy water', 'stop bleeding using pressure', 'salt baths', 'Consult a doctor'],
'GERD':['Avoid fatty spicy food', 'Avoid lying down after eating', 'Maintain healthy weight', 'Exercise'],
'Chronic cholestasis': ['Cold baths', 'Anti itch medicine', 'Eat Healthy', 'Consult a doctor'],
'hepatitis A':['Consult nearest hospital', 'Aash hands through', 'Avoid fatty spicy food'],
'Osteoarthristis': ['Acetaminophen', 'consult nearest hospital', 'follow up',	'salt baths'],
'(vertigo) Paroymsal Positional Vertigo':['lie down'	'avoid sudden change in body'	'avoid abrupt head movment'	'Relax'],
'Hypoglycemia': ['lie down on side',	'check in pulse',	'drink sugary drinks',	'consult doctor'],
'Acne':['bath twice',	'avoid fatty spicy food',	'drink plenty of water',	'avoid too many skincare products'],
'Diabetes':	['Have balanced diet',	'exercise',	'consult doctor',	'follow up'],
'Impetigo':['soak affected area in warm water', 'use antibiotics',	'remove scabs with wet compressed cloth','consult doctor'],
'Hypertension':	['meditation'	,'salt baths',	'reduce stress',	'get proper sleep'],
'Peptic ulcer diseae':['avoid fatty spicy food',	'consume probiotic food',	'eliminate milk',	'limit alcohol'],
'Dimorphic hemmorhoids(piles)':['avoid fatty spicy food',	'consume witch hazel',	'warm bath with epsom salt','consume alovera juice'],
'Common Cold':	['drink vitamin c rich drinks',	'take vapour',	'avoid cold food',	'keep fever in check'],
'Chicken pox':	['use neem in bathing','consume neem leaves',	'take vaccine',	'avoid public places'],
'Cervical spondylosis':	['use heating pad or cold pack',	'Exercise',	'take otc pain reliver',	'consult doctor'],
'Hyperthyroidism':	['Eat healthy','massage using lemon balm',	'take radioactive iodine treatment'],
'Urinary tract infection':	['drink plenty of water',	'increase vitamin c intake',	'drink cranberry juice',	'take probiotics'],
'Varicose veins':['lie down flat and raise the leg high',	'use oinments',	'use vein compression',	'Dont stand still for long'],
'AIDS'	:['avoid open cuts', 'wear ppe if possible',	'consult doctor',	'follow up'],
'Paralysis (brain hemorrhage)':	['massage'	,'eat healthy', 'exercise',	'consult doctor'],
'Typhoid':	['eat high calorie vegitables',	'antiboitic therapy',	'consult doctor',	'medication'],
'Hepatitis B':['consult nearest hospital',	'vaccination','eat healthy',	'medication'],
'Fungal infection':['bath twice',	'use detol or neem in bathing water',	'keep infected area dry',	'use clean cloths'],
'Hepatitis C': ['Consult nearest hospital',	'vaccination',	'eat healthy',	'medication'],
'Migraine'	:['meditation',	'reduce stress',	'use poloroid glasses in sun',	'consult doctor'],
'Bronchial Asthma':	['switch to loose cloothing',	'take deep breaths','get away from trigger','seek help'],
'Alcoholic hepatitis':	['stop alcohol consumption',	'consult doctor',	'medication',	'follow up'],
'Jaundice':	['drink plenty of water',	'consume milk thistle',	'eat fruits and high fiberous food',	'medication'],
'Hepatitis E':	['stop alcohol consumption',	'rest',	'consult doctor',	'medication'],
'Dengue':	['drink papaya leaf juice',	'avoid fatty spicy food',	'keep mosquitos away',	'Stay hydrated'],
'Hepatitis D':	['consult doctor','medication',	'eat healthy'	,'follow up'],
'Heart attack':['Call ambulance	', 'chew or swallow asprin',	'keep calm'],
'Pneumonia'	:['consult doctor',	'medication	rest'	'follow up'],
'Arthritis'	:['exercise',	'use hot and cold therapy',	'try acupuncture',	'massage'],
'Gastroenteritis'	:['stop eating solid food for a while',	'try taking small sips of water',	'rest',	'ease back into eating'],
'Tuberculosis':	['cover mouth','consult doctor',	'medication',	'rest']
 }


st.sidebar.title("SMARTHEALTH PROJECT")

st.title('Predict Disease from symptoms')
st.write("Indicate the Symptoms you are experiencing by selecting YES or NO")

input_vector = np.zeros(len(symptoms_dict))
user_symptoms = []
a = st.selectbox("visual_disturbances", ("NO","YES"))
if a == "YES":
    user_symptoms.append(symptoms_dict['visual_disturbances'])
b = st.selectbox("movement_stiffness", ("NO","YES"))
if b == "YES":
    user_symptoms.append(symptoms_dict['movement_stiffness'])

c = st.selectbox("mucoid_sputum", ("NO","YES"))
if c == "YES":
    user_symptoms.append(symptoms_dict['mucoid_sputum'])
d = st.selectbox("increased_appetite", ("NO","YES"))
if d == "YES":
    user_symptoms.append(symptoms_dict['increased_appetite'])
e = st.selectbox("spotting_ urination", ("NO","YES"))
if e == "YES":
    user_symptoms.append(symptoms_dict['spotting_ urination'])
f = st.selectbox("slurred_speech", ("NO","YES"))
if f == "YES":
    user_symptoms.append(symptoms_dict['slurred_speech'])
g = st.selectbox("unsteadiness", ("NO","YES"))
if g == "YES":
    user_symptoms.append(symptoms_dict['unsteadiness'])
h = st.selectbox("receiving_blood_transfusion", ("NO","YES"))
if h == "YES":
    user_symptoms.append(symptoms_dict['receiving_blood_transfusion'])
i = st.selectbox("blood_in_sputum", ("NO","YES"))
if i == "YES":
    user_symptoms.append(symptoms_dict['blood_in_sputum'])
j = st.selectbox("pain_behind_the_eyes", ("NO","YES"))
if j == "YES":
    user_symptoms.append(symptoms_dict['pain_behind_the_eyes'])
k = st.selectbox("rusty_sputum", ("NO","YES"))
if k == "YES":
    user_symptoms.append(symptoms_dict['rusty_sputum'])
l = st.selectbox("throat_irritation", ("NO","YES"))
if l == "YES":
    user_symptoms.append(symptoms_dict['throat_irritation'])
m = st.selectbox("weakness_in_limbs", ("NO","YES"))
if m == "YES":
    user_symptoms.append(symptoms_dict['weakness_in_limbs'])
n = st.selectbox("swelling_of_stomach", ("NO","YES"))
if n == "YES":
    user_symptoms.append(symptoms_dict['swelling_of_stomach'])
o = st.selectbox("pain_during_bowel_movements", ("NO","YES"))
if o == "YES":
    user_symptoms.append(symptoms_dict['pain_during_bowel_movements'])
p = st.selectbox("cramps", ("NO","YES"))
if p == "YES":
    user_symptoms.append(symptoms_dict['cramps'])
q = st.selectbox("skin_peeling", ("NO","YES"))
if q == "YES":
    user_symptoms.append(symptoms_dict['skin_peeling'])
r = st.selectbox("lack_of_concentration", ("NO","YES"))
if r == "YES":
    user_symptoms.append(symptoms_dict['lack_of_concentration'])
s = st.selectbox("red_spots_over_body", ("NO","YES"))
if s == "YES":
    user_symptoms.append(symptoms_dict['red_spots_over_body'])
t = st.selectbox("ulcers_on_tongue", ("NO","YES"))
if t == "YES":
    user_symptoms.append(symptoms_dict['ulcers_on_tongue'])

# loading in the model to predict on the data
input_vector =  input_vector[user_symptoms]
pickle_in = open('pickle_rawmodel.pkl', 'rb')
classifier = pickle.load(pickle_in)
if st.button('Predict Disease'):
    if len(input_vector)>0:
        classifier.predict_proba([input_vector])
        result = classifier.predict([input_vector])
        st.write("These are symptoms of {}".format(result[0]))
        st.title("Recommended Precautions:")
        if result[0] in precation_dict:
            for prec in precation_dict[result[0]]:
                st.write(prec)
    else:
        st.write("You are healthy")




