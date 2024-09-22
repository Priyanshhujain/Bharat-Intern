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



# Titanic Survivor Prediction

This project builds a **machine learning model** to predict whether a passenger survived the Titanic disaster based on several features. The dataset used is the famous Titanic dataset available from Kaggle.

## Project Structure
- `Titanic_survivor_prediction.ipynb`: Jupyter notebook containing code for data preprocessing, model building, training, and evaluation.
- `data`: Directory for storing the dataset (`train.csv`, `test.csv`).
- `models`: Directory to store trained model files for later use.

## Dataset
The dataset contains the following key features:
- **PassengerId**: ID of the passenger.
- **Pclass**: Ticket class (1st, 2nd, 3rd).
- **Name**: Name of the passenger.
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Ticket**: Ticket number.
- **Fare**: Passenger fare.
- **Cabin**: Cabin number (if available).
- **Embarked**: Port of embarkation.
- **Survived**: Target variable (1 if survived, 0 otherwise).

## Steps Involved
1. **Data Preprocessing**:
   - Handling missing values for features like Age, Cabin, and Embarked.
   - Encoding categorical features (e.g., Sex, Embarked).
   - Feature scaling for models sensitive to feature magnitude.

2. **Model Training**:
   - Several machine learning models are used, including Logistic Regression, Random Forest, and Decision Tree.
   - Cross-validation and hyperparameter tuning are performed to improve model performance.

3. **Evaluation**:
   - Accuracy, Precision, Recall, F1-score, and AUC are used for model evaluation.

## Results
The model achieves an accuracy of approximately `95%` (replace with actual result) in predicting whether a passenger survived the Titanic disaster. Precision, Recall, and AUC are also considered for evaluating model performance.



