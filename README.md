Web app using NLP and Machine Learning Pipeline
Part of the Udacity Data Science Nanodegree, the Disaster Response Project.

A web app is created with Flask and Bootstrap for  Disaster Response Project.

A new message is classified into categories – like 'aid related', 'search and rescue', 'child alone', and 'water' – based on the learnings from the labeled training data which contains real messages that were sent during disaster events

New training data can be provided and used to update the model. More precisely, data cleaning and storing in a database can be performed using an ETL pipeline, and training the classifier and providing the best model to the web app can be performed using a Machine Learning (ML) pipeline.

Requirements
Python 3 mainly with the packages Pandas, flask, plotly, nltk and sqlalchemy to run the web app. To use the pipelines mainly numpy and sklearn are needed.

Instructions

To run the pipelines:

Run the ETL pipeline via the command line in the data folder:
python python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
Run the ML pipeline via the command line in the models folder. The best model and its score are printed:
python python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

To run the web app:

Execute the Python file 'run.py' in the 'app' folder via the command line: python run.py
Go to http://0.0.0.0:3001/ in a browser.

Files

data contains the ETL pipeline (process_data.py) and the CSV input files plus the ETL pipeline output, an SQLite database.

models contains the ML pipeline (train_classifier.py) with its output, i.e. a Python pickle file with the best model from testing different classifiers and parameters. That pickle file is used in the app to classify new messages. and a Python pickle file with input for some of the graphs in the app.
