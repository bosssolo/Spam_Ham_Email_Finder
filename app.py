import streamlit as st 
from joblib import load
import numpy as np 
import re
import string

tfidf = load('NLP/tfidf.pkl')

model = load('NLP/Modle.pkl')


from sklearn.pipeline import Pipeline

def clean_emails(email_clean):
     email_clean = email_clean.lower() # Lowercase all characters 
     email_clean = re.sub(r'\d+','',email_clean) # Remove Digits 
     email_clean = email_clean.translate(str.maketrans('','',string.punctuation)) # Remove Punctutions
     email_clean = email_clean.replace('subject','') # remove Subject  
     email_clean = email_clean.strip() # reoving all extra spaces the beginning and ending 

     from nltk.stem import PorterStemmer , WordNetLemmatizer 
     # le = WordNetLemmatizer()
     # nltk.download('wordnet')
     # ' '.join([st.stem(word) for word in email_clean.split()])

     st = PorterStemmer()
     email_clean = ' '.join([st.stem(word) for word in email_clean.split()])
     return email_clean





st.title("Email Checker Spam or Ham Email")
st.write("Please give you email to check for its a spam or ham")

email = st.text_input("Enter you Email: ")

st.write('My_entered_Email: ', email)



if st.button("Check Email"):
    if email.strip() == "":
        st.warning("Please enter an email or message to check.")
    else:
        cleaned = clean_emails(email)
        x_input = tfidf.transform([cleaned])
        prediction = model.predict(x_input)
        if prediction == 1:
            st.error("Danger! This is a Spam Emails")
        
        else:
             st.success("All Good,This is a Genuine Emails")
            

        