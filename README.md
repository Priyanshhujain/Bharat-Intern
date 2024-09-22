# SMS Classifier

This project is a **machine learning model** that classifies SMS messages as **spam** or **ham (not spam)**. The dataset used for training the model consists of labeled SMS messages. The classifier is trained using various NLP techniques and algorithms.

## Project Structure
- `SMS_classifier.ipynb`: Jupyter notebook containing code for data preprocessing, training, testing, and evaluating the model.
- `data`: Folder containing the dataset used for training the model (e.g., spam.csv).
- `models`: Directory to store trained model files for later use.

## Dataset
The dataset contains two columns:
- **Message**: The text message.
- **Label**: The classification label ("ham" for non-spam, "spam" for spam messages).

## Steps Involved
1. **Data Preprocessing**:
   - Text cleaning (removing punctuation, stopwords, etc.)
   - Tokenization and vectorization (using TF-IDF, CountVectorizer, etc.)

2. **Model Training**:
   - Models like Naive Bayes, Support Vector Machine (SVM), or Logistic Regression are trained on the dataset.
   
3. **Evaluation**:
   - Accuracy, Precision, Recall, F1-score metrics are used to evaluate the model.

## Results
The classifier achieves an accuracy of approximately `84%` (replace with actual result). Other metrics such as Precision, Recall, and F1-Score are used for a more comprehensive performance evaluation.


