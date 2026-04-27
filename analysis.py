import pickle 
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import Counter
import os
from collections import Counter
# only for 2024
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
    pupil_mapping = dict(zip(county_data['School Name'],county_data['Pupil/Teacher Ratio [Public School] 2024-25']))
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
    data.to_csv(f"{TRAINING_DATA_FILES}/{year}-data.csv")








if __name__ == "__main__":
    # input each school data years from a directory
    TRAINING_DATA_DIRECTORY = "training-data"

    data_files = [f for f in os.listdir(TRAINING_DATA_DIRECTORY)]
    relevant_files = data_files[2:]
    for file in relevant_files:
        print(f"\n--- Processing: {file} ---")
        df = pd.read_csv(f"{TRAINING_DATA_DIRECTORY}/{file}")
        print(df['Area Type'].unique())




