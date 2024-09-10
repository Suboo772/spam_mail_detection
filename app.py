import streamlit as st
import pickle

# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
mnb = pickle.load(open('model.pkl', 'rb'))

# Define the prediction function
def predict_spam(sample_message):
    vector_input = tfidf.transform([sample_message])
    result = mnb.predict(vector_input)[0]
    if result == 1:
        return "Spam"
    else:
        return "Ham"

# Streamlit app
st.title("Spam Mail Detection Using ML")

# Get user input
user_input = st.text_area("Enter your message here:")

# Make prediction and display result
if st.button("Detect"):
    if user_input:
        prediction = predict_spam(user_input)
        st.write("Prediction:", prediction)
    else:
        st.write("Please enter a message.")