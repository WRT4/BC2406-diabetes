# Install libraries
pip install numpy
pip install pandas
pip install sklearn
pip install imblearn

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
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/diabetes_data_upload.csv")
# Remove duplicates
data_cleaned = df.drop_duplicates()

# Encode categorical variables
label_encoder = LabelEncoder()

# Store original gender labels for post-SMOTE inspection
gender_encoder = LabelEncoder()
data_cleaned['Gender'] = gender_encoder.fit_transform(data_cleaned['Gender'])

for column in data_cleaned.columns:
    if data_cleaned[column].dtype == 'object' and column != 'Gender':  # Skip encoding 'Gender' as we encoded it manually
        data_cleaned[column] = label_encoder.fit_transform(data_cleaned[column])

# Separate Female and Male data
female_data = data_cleaned[data_cleaned['Gender'] == gender_encoder.transform(['Female'])[0]]
male_data = data_cleaned[data_cleaned['Gender'] == gender_encoder.transform(['Male'])[0]]

# Separate features (X) and target (y) for SMOTE
X_female = female_data.drop('class', axis=1)
y_female = female_data['class']

# Apply SMOTE to only the Female group
smote = SMOTE(random_state=9)
X_resampled_female, y_resampled_female = smote.fit_resample(X_female, y_female)


# Recombine Female and Male data
X_resampled = pd.concat([X_resampled_female, male_data.drop('class', axis=1)], axis=0)
y_resampled = pd.concat([y_resampled_female, male_data['class']], axis=0)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=9)

# Initialize and train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Initialize the DecisionTreeClassifier (CART)
cart_model = DecisionTreeClassifier(max_depth=2, random_state=9)

# Train the model on the training data
cart_model.fit(X_train, y_train)

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=9)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Function to collect user input
def get_yes_no_input(prompt):
    response = st.radio(prompt, ('No', 'Yes'))
    return 1 if response == 'Yes' else 0

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

st.subheader("Please answer these questions to the best of your ability. If you are unsure, visit the doctor.")

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
responses["partial paresis"] = get_yes_no_input("Do you have partial paralysis where you are unable to control some muscles?")
responses["muscle stiffness"] = get_yes_no_input("Do you have muscle stiffness or a feeling of pain or tightness in your muscles?")
responses["Alopecia"] = get_yes_no_input("Do you have Alopecia or hair loss?")
responses["Obesity"] = get_yes_no_input("Do you have obesity?")

# Define column headings
columns = ["Age", "Gender"] + questions

# Convert responses dictionary to a DataFrame with column headings
response_df = pd.DataFrame([responses], columns=columns)

# Display the collected data (optional)
st.write("Collected Responses:", response_df)

# Function to interpret model results
def resultstring(arr):
    if arr[0] == 1:
        return "Positive"
    elif arr[0] == 0:
        return "Negative"
    else:
        return "Error"

# Simulate model predictions (replace these with actual model predictions)
# Predict with Logistic Regression
log_reg_pred = log_reg.predict(response_df)

# Predict with Decision Tree
cart_pred = cart_model.predict(response_df)

# Predict with Random Forest
rf_pred = rf_model.predict(response_df)

# Display predictions
st.subheader("Predictions")
st.write("Logistic Regression Prediction: " + resultstring(log_reg_pred))
st.write("Decision Tree Prediction: " + resultstring(cart_pred))
st.write("Random Forest Prediction: " + resultstring(rf_pred))

# If any model predicts positive, recommend visiting a doctor
num_of_positve = log_reg_pred[0] + cart_pred[0] + rf_pred[0]
if num_of_positve > 1:
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
