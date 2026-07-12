import streamlit as st
import pandas as pd
import pickle
import os

MODEL_PATH = os.path.join("model", "marks_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at:\n{MODEL_PATH}\nPlease run train_model.py first.")
else:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    st.set_page_config(page_title="Student Marks Predictor", layout="wide")

    st.markdown("""
        <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1523050854058-8df90110c9f1?auto=format&fit=crop&w=1950&q=80');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .header {
            text-align: center;
            padding: 30px;
            border-radius: 15px;
            background: rgba(75, 0, 130, 0.6);
            margin-bottom: 25px;
            color: #fff;
        }
        .input-card {
            background-color: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(8px);
            color: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }
        .prediction-card {
            background: linear-gradient(90deg, #8e44ad, #9b59b6);
            color: white;
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            margin-top: 20px;
        }
        .suggestion {
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            font-weight: bold;
            font-size: 18px;
        }
        .excellent { background-color: #D4EDDA; color: #155724; }
        .good { background-color: #FFF3CD; color: #856404; }
        .average { background-color: #FFE5B4; color: #663C00; }
        .below { background-color: #F8D7DA; color: #721C24; }
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #8e44ad, #9b59b6);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 30px;
            font-size: 18px;
            font-weight: bold;
            transition: 0.3s;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(90deg, #9b59b6, #8e44ad);
            transform: scale(1.03);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="header"><h1>🎓 Student Marks Predictor</h1><p>Predict marks for one or many students at once!</p></div>',
        unsafe_allow_html=True
    )

    
    option = st.radio("Select Prediction Mode", [" Single Student", " Multiple Students (CSV Upload)"])

   
    if option == " Single Student":
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🧾 Enter Student Details")

        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
            attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
            courses_registered = st.number_input("Courses Registered", min_value=0, max_value=10, value=5, step=1)

        with col2:
            previous_cgpa = st.number_input("Previous Year CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.01)
            social_media_usage = st.number_input("Social Media Usage (hrs/day)", min_value=0.0, max_value=12.0, value=2.0, step=0.5)

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Predict Marks"):
            input_df = pd.DataFrame([[study_hours, attendance, previous_cgpa, social_media_usage, courses_registered]],
                                    columns=["Study_Hours", "Attendance", "Previous_Year_CGPA", "Social_Media_Usage", "Courses_Registered"])
            predicted_marks = model.predict(input_df)[0]

            st.markdown(f'<div class="prediction-card">Predicted Marks: {predicted_marks:.2f}</div>', unsafe_allow_html=True)

            if predicted_marks >= 85:
                st.markdown('<div class="suggestion excellent">🎯 Excellent! Keep it up!</div>', unsafe_allow_html=True)
            elif 70 <= predicted_marks < 85:
                st.markdown('<div class="suggestion good">📘 Good job! Revise key topics.</div>', unsafe_allow_html=True)
            elif 50 <= predicted_marks < 70:
                st.markdown('<div class="suggestion average">⚠️ Average. Increase study hours!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="suggestion below">❌ Below average. Focus more on weak areas.</div>', unsafe_allow_html=True)

    
    else:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader(" Upload CSV File with Multiple Students")

        st.write("Your CSV file should contain columns:")
        st.code("Study_Hours, Attendance, Previous_Year_CGPA, Social_Media_Usage, Courses_Registered", language="text")

        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            if st.button("Predict Marks for All Students"):
                try:
                    predictions = model.predict(df)
                    df["Predicted_Marks"] = predictions
                    st.success(f"✅ Predicted marks for {len(df)} students successfully!")
                    st.dataframe(df)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Predictions as CSV", csv, "predicted_marks.csv", "text/csv")

                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
