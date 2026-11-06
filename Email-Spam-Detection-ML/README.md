# Email Spam Detection Using Machine Learning

## Project overview
This beginner-friendly project classifies a text message as **spam** or **ham (not spam)**.

## Technologies
- Python and Pandas
- Scikit-learn
- TF-IDF text vectorization
- Multinomial Naive Bayes
- Streamlit

## Files
- `app.py` — trains the model and runs the interactive app.
- `spam.csv` — small educational sample dataset.
- `requirements.txt` — required Python packages.

## Run in VS Code
1. Extract the ZIP and open the extracted folder in VS Code.
2. Open Terminal > New Terminal.
3. (Recommended) Create and activate a virtual environment:

   Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   python -m streamlit run app.py
   ```
6. Open the `Local URL` shown in the terminal (usually `http://localhost:8501`).

## How it works
1. Load labelled messages from the CSV file.
2. Split the dataset into training and testing data.
3. Convert text into numeric features using TF-IDF.
4. Train a Multinomial Naive Bayes classifier.
5. Predict whether a new message is spam or ham.

## Dataset limitation
The included dataset has 50 sample messages and is only for learning and demonstrating the workflow. It is not a representative real-world spam dataset. Report the measured test result honestly; do not claim production-level accuracy.

## GitHub submission
Create a public repository named `Email-Spam-Detection-ML`, upload the files, and submit the repository URL. Never upload passwords, API keys, or private information.
