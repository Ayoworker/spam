import streamlit as st
import joblib
import pandas as pd
import random

# Load the serialized model
model = joblib.load("spamnotification/Spam_detection_pipeline.pkl")

# Load the CSV file containing messages
messages_df = pd.read_csv("spamnotification/new_spam.csv")

# Function to detect spam
def detect_spam(message):
    # Use the loaded model to predict
    prediction = model.predict([message])[0]
    return prediction

# Function to generate a random message from the CSV file
def generate_random_message():
    random_index = random.randint(0, len(messages_df) - 1)
    random_message = messages_df.iloc[random_index]["v2"]
    return random_message

# Main function to run the app
def main():
    # Set the title of the web app
    st.title("Spam Detector")

    # Initialize session state
    if "message" not in st.session_state:
        st.session_state.message = ""

    # Text input for user message
    message = st.text_input("Enter your message:", value=st.session_state.message)

    # Generate Random Message button
    if st.button("Generate Random Message"):
        random_message = generate_random_message()
        st.session_state.message = random_message

    # Detect spam when the user presses Enter or clicks the button
    if st.button("Detect"):
        if message:
            # Detect spam
            prediction = detect_spam(message)

            # Map prediction to boolean value
            is_spam = (prediction == "spam")

            # Display prediction
            st.write(f"Prediction: {prediction}")

            # Display result
            if is_spam:
                st.error("It's likely a scam")
            else:
                st.success("You are safe")

# Run the app
if __name__ == "__main__":
    main()
