import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ------------------------------
# 1. Load and Prepare Data
# ------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("water_data.csv")

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, X.columns

model, scaler, features = load_model()

# ------------------------------
# 2. Streamlit UI
# ------------------------------
st.set_page_config(page_title="💧 Water Quality Prediction", layout="centered")

# Main Header
st.markdown(
    """
    <h1 style="text-align:center; color:#1d3557; font-size:50px; margin-bottom:10px;">
        🌊 Welcome to <span style="color:#0077b6;">AquaVision</span>
    </h1>
    <br><br>
    """,
    unsafe_allow_html=True
)

# App Title + Description
st.markdown(
    """
    <h1 style="text-align: center; color:#0077b6;">💧 AquaVision </h1>
    <p style="text-align: center; font-size:18px;">
    This app predicts whether the water is <b>Drinkable ✅</b> or <b>Not Drinkable ❌</b><br>
    using a trained <b>Random Forest Machine Learning Model</b>.
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar Input
st.sidebar.header("🔹 Enter Water Parameters")

user_input = []
for feature in features:
    val = st.sidebar.number_input(f"{feature}", value=0.0, format="%.4f")
    user_input.append(val)

user_input = np.array(user_input).reshape(1, -1)
user_scaled = scaler.transform(user_input)

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(user_scaled)[0]

    if prediction == 1:
        st.markdown(
            """
            <div style="padding:20px; border-radius:10px; background-color:#d4f7d4; text-align:center;">
                <h2 style="color:green;">✅ The water is Drinkable</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="padding:20px; border-radius:10px; background-color:#ffd6d6; text-align:center;">
                <h2 style="color:red;">❌ The water is Not Drinkable</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray; font-size:14px;">
    Built with ❤️ , Thanks for choosing AquaVision
    </p>
    """,
    unsafe_allow_html=True,
)

