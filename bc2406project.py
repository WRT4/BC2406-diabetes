# Import the libraries
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC


# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("data/diabetes_data_upload.csv")
    df2 = pd.read_csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

    # Remove duplicates and encode categorical variables
    data_cleaned = df.drop_duplicates()
    label_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()
    data_cleaned['Gender'] = gender_encoder.fit_transform(data_cleaned['Gender'])

    for column in data_cleaned.columns:
        if data_cleaned[column].dtype == 'object' and column != 'Gender':
            data_cleaned[column] = label_encoder.fit_transform(data_cleaned[column])

    # Apply SMOTE to only the Female group
    female_data = data_cleaned[data_cleaned['Gender'] == gender_encoder.transform(['Female'])[0]]
    male_data = data_cleaned[data_cleaned['Gender'] == gender_encoder.transform(['Male'])[0]]
    X_female = female_data.drop('class', axis=1)
    y_female = female_data['class']

    categorical_features = list(range(1, 16, 1))
    # Apply SMOTE-NC to only the Female group
    # Initialize SMOTE-NC with categorical feature indices
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=9)
    X_resampled_female, y_resampled_female = smote_nc.fit_resample(X_female, y_female)

    # Recombine Female and Male data
    X_resampled = pd.concat([X_resampled_female, male_data.drop('class', axis=1)], axis=0)
    y_resampled = pd.concat([y_resampled_female, male_data['class']], axis=0)

    # Process second dataset
    numerical_columns = ['BMI', 'MentHlth', 'PhysHlth']
    df2[df2.columns.difference(numerical_columns)] = df2[df2.columns.difference(numerical_columns)].astype("category")
    unimportant_variables = ['Smoker', 'Fruits', 'Stroke', 'AnyHealthcare', 'CholCheck', 'NoDocbcCost', 'Veggies', 'Education']
    df2.drop(columns=unimportant_variables, inplace=True)

    X_train3 = df2[df2.columns.difference(["Diabetes_binary"])]
    y_train3 = df2["Diabetes_binary"]

    return X_resampled, y_resampled, X_train3, y_train3, gender_encoder


# Cache model training
@st.cache_resource
def train_models(X_resampled, y_resampled, X_train3, y_train3):
    # Train logistic regression, decision tree, and random forest models
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_resampled, y_resampled)

    cart_model = DecisionTreeClassifier(max_depth=4, random_state=9)
    cart_model.fit(X_resampled, y_resampled)

    rf_model = RandomForestClassifier(random_state=9)
    rf_model.fit(X_resampled, y_resampled)

    cart_model2 = DecisionTreeClassifier(max_depth=7, random_state=9)
    cart_model2.fit(X_train3, y_train3)

    rf_model2 = RandomForestClassifier(max_depth=9, random_state=9)
    rf_model2.fit(X_train3, y_train3)

    return log_reg, cart_model, rf_model, cart_model2, rf_model2


# Load data and train models only once
X_resampled, y_resampled, X_train3, y_train3, gender_encoder = load_and_preprocess_data()
log_reg, cart_model, rf_model, cart_model2, rf_model2 = train_models(X_resampled, y_resampled, X_train3, y_train3)


# Function to collect user input
def get_yes_no_input(prompt):
    response = st.radio(prompt, ('No', 'Yes'))
    return 1 if response == 'Yes' else 0


def get_yes_no_input2(prompt):
    response = st.radio(prompt, ('No', 'Yes', "Not sure"))
    if response == 'Yes':
        return 1
    elif response == 'No':
        return 0
    else:
        return np.nan


# Title for the Streamlit App
st.title("Diabetes Prediction")
image1_link = "https://img.freepik.com/free-vector/cartoon-infographic-presenting-information-about-diabetes-symptoms-treatment-prevention_1284-53864.jpg?t=st=1729670751~exp=1729674351~hmac=20daa20302c3aa16c4604e7ae751367144eaf80393831e0c840c45a168ec0b13&w=740"
st.image(image1_link)

# Input fields for age and gender

age = st.number_input("What is your age this year?", min_value=0, max_value=120, step=1)

gender = st.selectbox("What is your sex assigned at birth?", options=["Male", "Female"])

# List of questions (yes/no)
questions = [
    "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
    "Polyphagia", "Genital thrush", "visual blurring", "Itching", "Irritability",
    "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity"
]

# Collect responses for the symptoms
responses = {"Age": age, "Gender": 1 if gender == 'Male' else 0}

if age >= 80:
    age_input = 13
elif 18 <= age < 80:
    age_input = (age - 18) // 5 + 1
else:
    age_input = np.nan
responses2 = {"Age": age_input, "Sex": 1 if gender == 'Male' else 0}

st.subheader("Please answer these questions to the best of your ability. If you are unsure, visit the doctor.")

height = st.number_input("What is your height in cm?", step=1, min_value=1) / float(100)
weight = st.number_input("What is your weight in kg?", step=0.1)
responses2["BMI"] = weight / (height * height)
responses2["HighBP"] = get_yes_no_input2("Do you have high blood pressure?")
responses2["HighChol"] = get_yes_no_input2("Do you have high cholestrol?")
gen = "Would you say that in general your health is:"
gen_options = {
    "Excellent": 1,
    "Very Good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5
}
selected_option1 = st.radio(label=gen, options=list(gen_options.keys()))
responses2["GenHlth"] = gen_options[selected_option1]
ment = "In the last 30 days, roughly how many days did you have poor mental health?"
responses2["MentHlth"] = st.number_input(label=ment, min_value=0, max_value=30, step=1)
phys = "In the last 30 days, roughly how many days did you have physical illness or injury?"
responses2["PhysHlth"] = st.number_input(label=phys, min_value=0, max_value=30, step=1)
responses2["DiffWalk"] = get_yes_no_input("Do you have serious difficulty walking or climbing stairs?")
# Define the options as a dictionary mapping description to integer


income_options = {
    "Less than $10,000": 1,
    "Less than $15,000 ($10,000 to less than $15,000)": 2,
    "Less than $20,000 ($15,000 to less than $20,000)": 3,
    "Less than $25,000 ($20,000 to less than $25,000)": 4,
    "Less than $35,000 ($25,000 to less than $35,000)": 5,
    "Less than $50,000 ($35,000 to less than $50,000)": 6,
    "Less than $75,000 ($50,000 to less than $75,000)": 7,
    "$75,000 or more": 8,
    "Don't know/Not sure": np.nan
}
selected_option2 = st.selectbox(label="Select your income level:", options=list(income_options.keys()))
responses2["Income"] = income_options[selected_option2]
responses2["HeartDiseaseorAttack"] = get_yes_no_input(
    "Do you have Coronary Heart Disease or have you ever had a heart attack?")
responses2["PhysActivity"] = get_yes_no_input("Have you done any physical activity in the past 30 days?")
responses2["HvyAlcoholConsump"] = get_yes_no_input(
    "Are you a heavy drinker? (>14 drinks per week for adult men and >7 drinks per week for adult women)")
responses["Polyuria"] = get_yes_no_input("Do you have Polyuria (excessive Urine Production)?")
responses["Polydipsia"] = get_yes_no_input("Do you have Polydipsia (excessive thirst)?")
responses["sudden weight loss"] = get_yes_no_input("Do you have sudden weight loss?")
responses["weakness"] = get_yes_no_input("Do you have weakness? Are you tired and low on energy often?")
responses["Polyphagia"] = get_yes_no_input("Do you have Polyphagia or extreme, insatiable hunger?")
responses["Genital thrush"] = get_yes_no_input("Have you gotten genital thrush or yeast infections down there before?")
responses["visual blurring"] = get_yes_no_input("Do you have blurry vision?")
responses["Itching"] = get_yes_no_input("Do you itch often thoughout the body, especially in the lower legs?")
responses["Irritability"] = get_yes_no_input("Do you have irritability and mood swings?")
responses["delayed healing"] = get_yes_no_input("Do your wounds heal slower than before?")
responses["partial paresis"] = get_yes_no_input(
    "Do you have partial paralysis where you are unable to control some muscles?")
responses["muscle stiffness"] = get_yes_no_input(
    "Do you have muscle stiffness or a feeling of pain or tightness in your muscles?")
responses["Alopecia"] = get_yes_no_input("Do you have Alopecia or hair loss?")
responses["Obesity"] = get_yes_no_input("Do you have obesity?")

# Define column headings
columns = ["Age", "Gender"] + questions
columns2 = list(X_train3.columns)


# Function to interpret model results
def resultstring(arr):
    if arr[0] == 1:
        return "Positive"
    elif arr[0] == 0:
        return "Negative"
    else:
        return "Error"


if st.button("Calculate"):
    # Convert responses dictionary to a DataFrame with column headings
    response_df = pd.DataFrame([responses], columns=columns)
    response2_df = pd.DataFrame([responses2], columns=columns2)
    # Display the collected data (optional)
    # st.write("Collected Responses:", response2_df)
    # Simulate model predictions (replace these with actual model predictions)
    # Predict with Logistic Regression
    log_reg_pred = log_reg.predict(response_df)
    # Predict probabilities for each class
    log_reg_probs = log_reg.predict_proba(response_df)

    # Predict with Decision Tree
    cart_pred = cart_model.predict(response_df)
    cart_pred_probs = cart_model.predict_proba(response_df)
    cart_pred2 = cart_model2.predict(response2_df)
    cart_pred_probs2 = cart_model2.predict_proba(response2_df)

    # Predict with Random Forest
    rf_pred = rf_model.predict(response_df)
    rf_pred_probs = rf_model.predict_proba(response_df)
    # Enforcing float format with one decimal place explicitly
    rf_pred_probs_formatted = ["{:.1f}".format(prob) for prob in np.round(rf_pred_probs[:, 1] * 100, 1)]

    rf_pred2 = rf_model2.predict(response2_df)
    rf_pred_probs2 = rf_model2.predict_proba(response2_df)
    # Enforcing float format with one decimal place explicitly
    rf_pred_probs_formatted2 = ["{:.1f}".format(prob) for prob in np.round(rf_pred_probs2[:, 1] * 100, 1)]

    avg_prob = (log_reg_probs[:, 1] * 100 + cart_pred_probs[:, 1] * 100 + rf_pred_probs[:, 1] * 100) / 3
    avg_prob2 = (cart_pred_probs2[:, 1] * 100 + rf_pred_probs2[:, 1] * 100) / 2
    avg_overall = (avg_prob + avg_prob2)/2
    
    # Display predictions
    st.header("Predictions")
    st.subheader("Dataset 1:")
    st.write("Logistic Regression Prediction: " + resultstring(log_reg_pred))
    st.write("Decision Tree Prediction: " + resultstring(cart_pred))
    st.write("Random Forest Prediction: " + resultstring(rf_pred))
    # st.write("There is a " + str(round(avg_prob[0], 1)) + "% chance of you having diabetes.")
    st.subheader("Dataset 2:")
    st.write("Decision Tree Prediction: " + resultstring(cart_pred2))
    st.write("Random Forest Prediction: " + resultstring(rf_pred2))
    # st.write("There is a " + str(round(avg_prob2[0], 1)) + "% chance of you having diabetes.")
    
    st.subheader("Overall prediction: ")
    st.write("There is a " + str(round(avg_overall[0], 1)) + "% chance of you having diabetes.")
    
    # If the majority of models predict positive, recommend visiting a doctor
    num_of_positve = log_reg_pred[0] + cart_pred[0] + rf_pred[0] + cart_pred2[0] + rf_pred2[0]
    if num_of_positve > 2:
        st.warning("Please visit the doctor!")
    else:
        st.success("No immediate concerns.")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #003866, #006455); /* Dark blue to dark green */
        height: 100vh; /* Full height */
        padding: 20px; /* Optional padding */
        color: white; /* Change text color for better visibility */
    }
    </style>
    """,
    unsafe_allow_html=True
)
