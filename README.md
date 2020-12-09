<h1>Web app using NLP and Machine Learning Pipeline</h1>
<h3>Part of the Udacity Data Science Nanodegree, the Disaster Response Project</h3>

A new message is classified into categories – like 'aid related', 'search and rescue', 'child alone', and 'water' – based on the learnings from the labeled training data which contains real messages that were sent during disaster events

New training data can be provided and used to update the model. More precisely, data cleaning and storing in a database can be performed using an ETL pipeline, and training the classifier and providing the best model to the web app can be performed using a Machine Learning (ML) pipeline.

<h2>Requirements</h2>
<br>Python 3 mainly with the packages Pandas, nltk and sqlalchemy to run the web app. To use the pipelines mainly numpy and sklearn are needed.</br>

<h2>Instructions</h2>

<h5>To run the pipelines:</h5>

1.Run the ETL pipeline via the command line in the data folder:
<br>python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2.Run the ML pipeline via the command line in the models folder. The best model and its score are printed:
<br>python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

<h5>To run the web app:</h5>

Execute the Python file 'run.py' in the 'app' folder via the command line: 
<br>python run.py
<br>Go to http://0.0.0.0:3001/ in a browser.

<h2>Files</h2>

data contains the ETL pipeline (process_data.py) and the CSV input files plus the ETL pipeline output, an SQLite database.

models contains the ML pipeline (train_classifier.py) with its output, i.e. a Python pickle file

app contains the web application.
