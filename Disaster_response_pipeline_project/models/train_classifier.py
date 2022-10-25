import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_data(database_filepath):
    """ load data from SQL Database 

    Input: database_filepath

    Output: X, Y and category_names for building ML model

    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('DisasterMessages', con=engine)

    # split data
    df.dropna(inplace=True)
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """process text input and  return cleaned tokens"""

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Build the  maching learning pipeline """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # improvement of accuray by hyperparameter tuning  
    parameters = {'tfidf__use_idf':(True, False),
              'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__min_samples_split': [2, 4],
              'vect__ngram_range': ((1, 1), (1, 2)),
              'clf__estimator__max_features': ['auto', 'sqrt']}

    cv= GridSearchCV(pipeline, param_grid=parameters,  cv=3, n_jobs=-1, verbose=1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    To evaluate the performance of the model

    Input: model to use, test dataset and category_names

    Output: Print out the accuracy of the prediction
    """

    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):

        print(classification_report(Y_test[col], y_pred[:, i]))


def save_model(model, model_filepath):
    """ Save model as picke file"""

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
