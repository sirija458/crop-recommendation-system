import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title
st.title("🌱 ML Crop Recommendation System")

st.write("Random Forest Algorithm with Accuracy & Graph")

# --------------------------
# STEP 1: Dataset
# --------------------------
data = {
    'N': [90,120,60,30,100,40,80,110,95,105,70,50],
    'P': [40,60,30,20,50,10,35,55,45,52,28,18],
    'K': [40,50,20,10,60,30,45,65,48,62,25,15],
    'rainfall': [200,100,150,50,180,70,160,120,210,130,140,60],
    'crop': ['Rice','Maize','Wheat','Cotton','Rice','Barley',
             'Wheat','Maize','Rice','Maize','Wheat','Cotton']
}

df = pd.DataFrame(data)

# --------------------------
# STEP 2: Split Data
# --------------------------
X = df[['N', 'P', 'K', 'rainfall']]
y = df['crop']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------
# STEP 3: Train Model
# --------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --------------------------
# STEP 4: Accuracy
# --------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader(f"📊 Model Accuracy: {accuracy*100:.2f}%")

st.info("Note: Accuracy may vary due to small dataset size.")

# --------------------------
# STEP 5: Graph (Crop Distribution)
# --------------------------
st.subheader("📈 Crop Distribution Graph")

crop_counts = df['crop'].value_counts()

fig, ax = plt.subplots()
ax.bar(crop_counts.index, crop_counts.values)
ax.set_xlabel("Crop")
ax.set_ylabel("Count")
ax.set_title("Crop Distribution in Dataset")

st.pyplot(fig)

# --------------------------
# STEP 6: User Input
# --------------------------
st.subheader("Enter Soil Details")

N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# --------------------------
# STEP 7: Prediction
# --------------------------
if st.button("Predict Crop"):
    input_data = [[N, P, K, rainfall]]
    prediction = model.predict(input_data)

    st.success(f"🌾 Recommended Crop: {prediction[0]}")