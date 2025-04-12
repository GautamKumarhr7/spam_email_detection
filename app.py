import streamlit as st
import pandas as pd
from spam_detector import SpamDetector

st.title("📧 Spam Email Identifier")

uploaded_file = st.file_uploader("Upload your email CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns or 'label' not in df.columns:
            st.error("CSV must contain 'label' and 'text' columns.")
        else:
            st.success("CSV uploaded successfully.")

            # Train model
            detector = SpamDetector()
            detector.train(df)

            # Predict on the same data (or you can upload separate test data)
            df['Prediction'] = detector.predict(df['text'])

            # Show the predictions
            spam_df = df[df['Prediction'] == 'spam']
            st.subheader("🛑 Detected Spam Emails")
            st.dataframe(spam_df[['text']])

            st.subheader("📊 Summary")
            st.write(df['Prediction'].value_counts())

    except Exception as e:
        st.error(f"An error occurred: {e}")
