import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Email Spam Detector", page_icon="📧")
st.title("📧 Email Spam Detection")
st.write("A beginner Machine Learning project using TF-IDF and Multinomial Naive Bayes.")

@st.cache_resource
def train_model():
    data = pd.read_csv("spam.csv").dropna(subset=["message", "label"])
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"].astype(str), data["label"].astype(str),
        test_size=0.25, random_state=42, stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return (model, accuracy_score(y_test, predictions),
            classification_report(y_test, predictions, zero_division=0),
            confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

model, accuracy, report, matrix = train_model()
st.metric("Test accuracy on included sample dataset", f"{accuracy:.1%}")
st.caption("This is a small teaching dataset; its accuracy is not a reliable measure of real-world performance.")

st.subheader("Try an email message")
message = st.text_area("Enter or paste an email message:", height=140,
                       placeholder="Example: Congratulations! You have won a free prize...")
if st.button("Predict", type="primary"):
    if message.strip():
        prediction = model.predict([message])[0]
        if prediction == "spam":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: NOT SPAM (HAM)")
        probabilities = model.predict_proba([message])[0]
        classes = model.named_steps["classifier"].classes_
        st.write("Model confidence scores:")
        st.json({str(c): round(float(p), 3) for c, p in zip(classes, probabilities)})
    else:
        st.warning("Please enter a message first.")

with st.expander("View model evaluation"):
    st.write("Classification report")
    st.code(report)
    st.write("Confusion matrix (rows = actual; columns = predicted)")
    st.dataframe(pd.DataFrame(matrix, index=["Actual ham", "Actual spam"],
                              columns=["Predicted ham", "Predicted spam"]))

with st.expander("About this project"):
    st.markdown("""
    - Dataset: `spam.csv`, with example messages labelled `spam` or `ham`.
    - Text processing: TF-IDF converts text into numerical features.
    - Algorithm: Multinomial Naive Bayes.
    - Interface: Streamlit.
    - Limitation: The included dataset is small and intended for learning.
    """)
