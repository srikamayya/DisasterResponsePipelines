import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    Read data from csv files
    
    Input : File names along with path
    Output: Complete data in DataFrame 
    """
    
    #Read data related to messages_filepath to messages variable
    messages = pd.read_csv(messages_filepath,encoding = 'UTF-8')
    
    #Read data related to categories_filepath to categories variable
    categories = pd.read_csv(categories_filepath, encoding = 'UTF-8')
    
    #Merge data sets
    data_df = pd.merge(messages, categories, on='id')   
    
    return data_df


def clean_data(df):
    
    """
    Cleaning data like split the data, removing duplicate data from Dataframe
    
    Input : DataFrame
    Output : DataFrame
    
    """
    
    print(" Spliting Categories data.... ")
    
    #split the data related to categories
    categories = df.categories.str.split(";",expand=True)
    
    #Read 1st row of data to assign them as column names
    row = categories[categories.index == 0]
    
    
    # extract a list of new column names for categories.
    category_colnames = pd.Series(row.values[0]).apply(lambda x: x[:-2]).tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, related-0 becomes 0, related-1 becomes 1
    for column in categories:
        categories[column] = pd.Series(categories[column].values).apply(lambda x: x[-1:])
        categories[column] = categories[column].astype('int64')
        
    if (categories.isna().sum().sum() > 0):
        print('error: issue with NaN values')
        sys.exit()
    
    # drop the original categories column from `df`
    df = df.drop(columns = 'categories', axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # removing invalid values for column 'related'
    df = df[df.related != 2]
    
    return df


def save_data(df, database_filename):
    
    """
    It will create .db file and stores the data in respective table in .db file
    
    Input: DataFrame and .db file name
    Output: NA
    
    """
    
    #Create engine to create db file
    engine = create_engine('sqlite:///'+database_filename)
    
    #Load data to db file
    df.to_sql('DisasterResponse', con = engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)


        print('Cleaning data...')
        df = clean_data(df)
        
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()