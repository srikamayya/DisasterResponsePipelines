import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    
    """
    load data based on database file 
    
    Input: Database file
    Output : Read data from table and assign it respective variable
              X : data related to message column
              Y : data relted to categories column(0-36)
              category_names : list of categories columns
    """
    
    #Create engine to connect to db file
    engine = create_engine('sqlite:///'+database_filepath)
    
    #Read data from table
    df = pd.read_sql_table('DisasterResponse', engine)
    
    #Assign data to variable
    category_names = df.columns.drop(['id','message','original','genre'])
    X = df.message.values
    Y = df[category_names]
    
    
    return X, Y, category_names


def tokenize(text):
    
    """
    Tokenize text messsages and cleaning for Machine Learning use.
    
    Input: str
    Output: list
    
    """
    
    # Replace all special characters
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    
    #Initialize the tokens word tokenize and lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    #cleaning the data
    clean_tokens = []
    for tok in tokens:
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    
    """
    Creating the pipeline and parameters
    
    Output: Molel pipeline
    
    """
    pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])

    

    parameters = {
    'clf__estimator__n_estimators': np.arange(16,32+1) 
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    
    """
    Based on eacch category it will calculate f1 score, precision and recall 
    
    Input : Model, X_test data, Y_test data, list of category_names
    Output:  f1 score, precision and recall 
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = category_names
    report = {}
    for col in y_pred_df.columns:
        output = classification_report(y_test[col], y_pred_df[col], output_dict=True)
        report[col] = {} # inspired by https://stackoverflow.com/questions/16333296/how-do-you-create-nested-dict-in-python
        for i in output:
            if i == 'accuracy':
                break
            report[col]['f1_' + i] = output[i]['f1-score']
            report[col]['precision_' + i] = output[i]['precision']
            report[col]['recall_' + i] = output[i]['recall']
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df[report_df.columns.sort_values()]
    report_df_mean = report_df.mean()
    print(report_df)
    
    return report_df, report_df_mean


def save_model(model, model_filepath):
    
    """
    Creating modle file in respective model_filepath
    
    Input : Model name and Filepath
    Output : NA
    """
    
    #Creating .pkl file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()