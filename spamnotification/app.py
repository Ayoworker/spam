import streamlit as st
import joblib

# Load the serialized model
model = joblib.load("spam_classification_pipeline.pkl")


# Function to detect spam
def detect_spam(message):
    # Use the loaded model to predict
    prediction = model.predict([message])[0]
    return prediction


# Main function to run the app
def main():
    # Set the title of the web app
    st.title("Spam Detector")

    # Get input message from the user
    message = st.text_input("Enter your message:")

    # Initialize is_spam variable
    is_spam = False

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
                st.write("It's likely a scam")
            else:
                st.write("You are safe")


# Run the app
if __name__ == "__main__":
    main()
