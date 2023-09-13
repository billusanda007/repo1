import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Import necessary libraries
import re
from urllib.parse import urlparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources
stop_words = set(stopwords.words("english"))  # Create a set of English stopwords
lemmatizer = WordNetLemmatizer()  # Initialize the WordNet Lemmatizer

# Define a function for text processing
def textProcess(sent):
    try:
        if sent is None:  # Check if the input is None
            return ""  # Return an empty string if input is None

        # Remove square brackets, parentheses, and other special characters
        sent = re.sub('[][)(]', ' ', sent)

        # Tokenize the text into words
        sent = [word for word in sent.split() if not urlparse(word).scheme]

        # Join the words back into a sentence
        sent = ' '.join(sent)

        # Remove Twitter usernames (words starting with @)
        sent = re.sub(r'\@\w+', '', sent)

        # Remove HTML tags using regular expression
        sent = re.sub(re.compile("<.*?>"), '', sent)

        # Remove non-alphanumeric characters (keep only letters and numbers)
        sent = re.sub("[^A-Za-z0-9]", ' ', sent)

        # Convert text to lowercase
        sent = sent.lower()

        # Split the text into words, strip whitespace, and join them back into a sentence
        sent = [word.strip() for word in sent.split()]
        sent = ' '.join(sent)

        # Tokenize the text again
        tokens = word_tokenize(sent)

        # Remove stop words
        for word in tokens.copy():
            if word in stop_words:
                tokens.remove(word)

        # Lemmatize the remaining words
        sent = [lemmatizer.lemmatize(word) for word in tokens]

        # Join the lemmatized words back into a sentence
        sent = ' '.join(sent)

        # Return the processed text
        return sent

    except Exception as ex:
        print(sent, "\n")
        print("Error ", ex)
        return ""  # Return an empty string in case of an error

# Rest of your code...

# Load the pre-trained model from joblib
model = joblib.load('Stress identification NLP')

# Load the TF-IDF vectorizer used during training
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Define the Streamlit web app
def main():
    st.title("Stress Predictor Web App")
    st.write("Enter some text to predict if the person is in stress or not.")

    # Input text box
    user_input = st.text_area("Enter text here:")

    if st.button("Predict"):
        if user_input:
            # Process the input text
            processed_text = textProcess(user_input)

            # Use the same TF-IDF vectorizer to transform the input text
            tfidf_text = tfidf_vectorizer.transform([processed_text])

            # Make predictions using the loaded model
            prediction = model.predict(tfidf_text)[0]

            if prediction == 1:
                result = "This person is in stress."
            else:
                result = "This person is not in stress."

            st.write(result)

if __name__ == '__main__':
    main()
