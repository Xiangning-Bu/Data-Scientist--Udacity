# Disaster Response Pipeline Project

### Instructions:

The project contains a web app which is able to classify disaster messages to speified categories. By applying data engineering skills on disater datasers from Figure Eight, we are able to build a model which could categorize messages on a real time basis.

There are three sections:
1. Build an ETL pipeline to extract data from the given datasets, clean the data, and then store it in a SQLite database
2. Create a machine learning pipeline to output a the model which could predict a message classifications
3. Develop a web application to show classify messages in real time

### Files description:
1. app
    - template
        - master.html # main page of web app
        - go.html # classification result page of web app
    - run.py # Flask file that runs app

2. data
    - disaster_categories.csv: Categories of the messages
    - disaster_messages.csv: Multilingual disaster response messages
    - process_data.py

3. models
    - train_classifier.py

4. README.md

### Running the code:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
