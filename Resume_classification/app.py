import streamlit as st
import pickle
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')

with open('new_model.pkl', 'rb') as file:
    model = pickle.load(file)

# model = pickle.load(open('new_model.pkl', 'rb'))
with open('vectorizer1.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


def cleanresume(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():

    st.markdown(
        f"""
            <style>
            .stApp {{
                background-image: url("https://cdn.pixabay.com/photo/2019/12/15/08/33/blue-4696575_960_720.jpg");
                background-attachment: fixed;
                background-size: cover
            }}
            </style>
            """,
        unsafe_allow_html=True
    )

    st.title("Resume Classification App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'docx', 'doc', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanresume(resume_text)
        input_feature = vectorizer.transform([cleaned_resume])
        prediction_id = model.predict(input_feature)[0]
        st.write(prediction_id)

        category_mapping = {
            0: 'Peoplesoft Developer',
            1: 'ReactJs Developer',
            2: 'SQL Developer',
            3: 'Workday'
        }

        if prediction_id in category_mapping:
            category_name = category_mapping[prediction_id]
            st.write("Predicted Category : ", category_name)
        else:
            st.write('Unknown Category')


if __name__ == "__main__":
    main()
