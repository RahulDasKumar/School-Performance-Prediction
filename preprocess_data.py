from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle

import torch 
import pandas as pd
import numpy as np
# from  sklearn.linear_model import LinearRegression
import re
import os
import re
import argparse

parser = argparse.ArgumentParser(description="A sample script.")
parser.add_argument("-p", "--procedure", type=str, help="Do you want to preprocess the files?", default="")
args = parser.parse_args()

SCHOOL_DIRECTORY = 'school-data'

SCHOOL_DATA = 'data/school-data.csv' # only for one year
COUNTY_DATA = 'data/county-information-all.csv' # only for one year (this can change)
COUNTY_SCHOOL_MAPPING = 'data/school-with-county.csv' 
CRIME_DATA ='data/crime_data.csv' # exists for every year from 2011 to current year
TRAINING_DATA_FILES = "training-data"
# reading the data







def data_merger(year:int,school_data=SCHOOL_DATA,county_information=COUNTY_DATA, crime_information = CRIME_DATA):
    """
    Args:
        year: used to filter which year of crime to extract from the crime data information
        school_data: information about the school performance -> each file is a year
        county_information: used as school-county-city mapping
        crime_information: contains the crime information for every county and city for each year 
    :rtype: pd.Dataframe
    """
    
    data = pd.read_csv(school_data, low_memory=False)
    county_data = pd.read_csv(county_information, on_bad_lines='skip')

    county_data['School Name'] = county_data['School Name'].apply(lambda x: str(x).lower())
    data['name'] = data['name'].apply(lambda x: str(x).lower())

    """
    creating the mappings for what we want the schools to be analyzed on
    """
    data_mapping = dict(zip(county_data['School Name'],county_data['County Name [Public School] 2024-25']))
    # figure out a way to change this column name based on the name
    year_to_string = str(year)
    last_two_digits = int(year_to_string[2:])
    next_year = last_two_digits + 1
    column_year = f"{year_to_string}-{next_year}"
    print("last two digits extraction: ", last_two_digits)
    print("column year: ",  column_year)
    pupil_mapping = dict(zip(county_data['School Name'],county_data[f'Pupil/Teacher Ratio [Public School] {column_year}']))
    city_mapping = dict(zip(county_data['School Name'],county_data['Location City [Public School] 2024-25']))

    """
    appending the matched schools that appear. Some schools such as county only schools tend to be missed
    """
    data['county'] = data['name'].apply(lambda x: data_mapping.get(x,"Not Found"))
    data['ratio'] = data['name'].apply(lambda x: pupil_mapping.get(x,"Not Found"))
    data['city'] = data['name'].apply(lambda x: city_mapping.get(x,"Not Found"))

    if not os.path.exists(COUNTY_SCHOOL_MAPPING):
        data.to_csv(COUNTY_SCHOOL_MAPPING)

        
    # ATTACH COUNTY DATA TO EACH SCHOOLS
    # we need to map each school with the county school rates and pupil issues
    # i can take a guess that schools with a bigger teacher pupil ratio will have higher crime rates
    # # for each school, look into the city it has mapped and the county 
    crime_data = pd.read_csv(crime_information)
    # filter to current year
    crime_data = crime_data[crime_data['Year']==year]
    # make all variables lower case for matching purposes

    crime_data['Area Name'] = crime_data['Area Name'].apply(lambda x: x.lower())
    data['city'] = data['city'].apply(lambda x: str(x).lower())
    crime_data['Area Name'] = crime_data['Area Name'].replace('charlotte-mecklenburg', 'charlotte')
    data['city_merge_key'] = data['city'].replace('charlotte', 'charlotte-mecklenburg')

    data = data.merge(
        crime_data,
        left_on='city_merge_key',
        right_on='Area Name',
        how='left'
    )
    # final data merged
    if not os.path.exists(TRAINING_DATA_FILES):
        os.makedirs(TRAINING_DATA_FILES)
    data = data.drop_duplicates(keep=False)

    data.to_csv(f"{TRAINING_DATA_FILES}/{year}-data.csv")



def preprocess_data(file_path:str):
    """
    file_name: filepath of the txt file, it has to be a txt file and tab delimited
    the default data is in a txt file, lets convert them to csv files
    """
    # preprocessing data 
    with open(file_path, 'r') as file:
        all_data = []
        line = file.readline()
        headers = re.split("\s+",line)
        # data has 15 columns total
        columns = headers[:15]
        for line in file.readlines():
            data = re.split("\t",line)
            # some lines are missing avg score, that will be my prediction value, will throw out the rows that dont have the same amount of features
            if len(data) >= 13:
                array = data[:15]
                all_data.append(array)
    print(len(columns))        

    dataset = np.array(all_data) 
    data_dict = dict()
    for index in range(len(columns)):
        data_dict[columns[index]] = dataset[:,index]
            
    print(data_dict)

    dataframe = pd.DataFrame(data_dict)
    return dataframe



def preprocess_crime_data(file_path):
    # preprocessing data 
    with open(file_path, 'r') as file:
        all_data = []
        line = file.readline()
        headers = re.split(";",line)
        columns = headers
        for line in file.readlines():
            data = re.split(";",line)
            all_data.append(data)
    print(len(columns))        

    dataset = np.array(all_data) 
    data_dict = dict()
    for index in range(len(columns)):
        data_dict[columns[index]] = dataset[:,index]
            
    print(data_dict)

    dataframe = pd.DataFrame(data_dict)
    return dataframe
    
    
def create_training_data():
    files = os.listdir(SCHOOL_DIRECTORY)
    # filter for csv files
    csv_files = [file for file in files if file.endswith(".csv")]
    print(csv_files)
    for file in csv_files:
        year = int(re.search(r'\d{4}',file).group())
        file_path = f"{SCHOOL_DIRECTORY}/{file}"
        data_merger(year,school_data=file_path)
        
def create_csv_files():
    files = os.listdir(SCHOOL_DIRECTORY)
    for file in files:
        year = re.search(r'\d{4}',file).group()
        file_path = f"{SCHOOL_DIRECTORY}/{file}"
    
        data = preprocess_data(file_path)
        # add the data into the same directory but for inputting the data for merging, lets only filter csv's
        data.to_csv(f'{SCHOOL_DIRECTORY}/school-data-{year}.csv')
        
if __name__ == "__main__":
    files = os.listdir(SCHOOL_DIRECTORY)
    if args.procedure == "file":
        create_csv_files()
    elif args.procedure == "training-data":
        create_training_data()
    else:
        create_csv_files()
        create_training_data()
        