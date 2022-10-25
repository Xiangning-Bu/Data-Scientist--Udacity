import sys

import pandas as pd 
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):

    """
    Load two dataset: messages and categories

    Input: filepath of two dataset

    Output: Merged dataframe based on id
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on =['id'])

    return df


def clean_data(df):
    """
    Data preparation/cleanup of combined dataframe

    Input: Dataframe df

    Output: cleaned dataframe
    """

    #split categories in to separate categoty columns
    categories = df['categories'].str.split(';', expand=True)

    #seletc first row and extract the column names
    category_colnames = categories.loc[0, :].apply(lambda x : x[:-2])

    #change column name
    categories.columns = category_colnames

    #convert category values to 0, 1 or 2
    for col in categories:
        categories[col]= categories[col].astype(str).str[-1].astype(int)
        
    #Drop rows whose "related" value is 2
    categories = categories[categories['related'] != 2]

    #replace categories column in df with new category colymns
    df.drop(columns=['categories'], inplace = True)
    df= pd.concat([df, categories], join='inner', axis =1) # use join=inner, otherwise int will change to float

    #remove duplates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """ Save the dataset to an sqlite database """

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')


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
