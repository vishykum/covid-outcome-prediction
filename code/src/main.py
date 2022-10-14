import pandas as pd
import numpy as np
from matplotlib.cbook import boxplot_stats  
from geopy.geocoders import Nominatim

cases_train = pd.read_csv("/Users/evancoulter/Desktop/Cmpt/Courses/cmpt459/cmpt-459-group-project/code/data/cases_2021_train.csv")
cases_test = pd.read_csv("/Users/evancoulter/Desktop/Cmpt/Courses/cmpt459/cmpt-459-group-project/code/data/cases_2021_test.csv")
location_2021 = pd.read_csv("/Users/evancoulter/Desktop/Cmpt/Courses/cmpt459/cmpt-459-group-project/code/data/location_2021.csv")

# ------------ 1.1 Cleaning Messy Labels ------------
if 'outcome' in cases_train.columns:
    conditions = [
        (cases_train['outcome'] == 'Discharged') | (cases_train['outcome'] == 'Discharged from hospital') | (cases_train['outcome'] == 'Hospitalized') | (cases_train['outcome'] == 'critical condition') | (cases_train['outcome'] == 'discharge') | (cases_train['outcome'] == 'discharged'),
        (cases_train['outcome'] == 'Alive') | (cases_train['outcome'] == 'Receiving Treatment') | (cases_train['outcome'] == 'Stable') | (cases_train['outcome'] == 'Under treatment') | (cases_train['outcome'] == 'recovering at home 03.03.2020') | (cases_train['outcome'] == 'released from quarantine') | (cases_train['outcome'] == 'stable') | (cases_train['outcome'] == 'stable condition'),
        (cases_train['outcome'] == 'Dead') | (cases_train['outcome'] == 'Death') | (cases_train['outcome'] == 'Deceased') | (cases_train['outcome'] == 'Died') | (cases_train['outcome'] == 'death') | (cases_train['outcome'] == 'died'),
        (cases_train['outcome'] == 'Recovered') | (cases_train['outcome'] == 'recovered')
    ]

    values = ['hospitalized', 'nonhospitalized', 'deceased', 'recovered']
    cases_train['outcome_group'] = np.select(conditions, values)
    del cases_train["outcome"]

    cases_train.to_csv("../data/cases_2021_train.csv")


# ------------ 1.4 Data Cleaning and Imputing Missing Values ------------
# Remove all entries with missing age values
cases_train_cleaned = cases_train.copy()
cases_test_cleaned = cases_test.copy()
cases_train_cleaned = cases_train_cleaned[cases_train_cleaned["age"].notna()]
cases_test_cleaned = cases_test_cleaned[cases_test_cleaned["age"].notna()]
# Correct format for werid formatted age entries
def format_wrong_ages(x):
    if "-" in x:
      numbers = x.split("-")
      if len(numbers[1]) != 0:
          new_age = str((int(numbers[1])+int(numbers[0]))/2)
          return int(new_age.split(".")[0])
      else:
        return int(numbers[0])
    elif "." in x:
      return int(x.split(".")[0])
    else:
      return int(x)
cases_train_cleaned["age"] = cases_train_cleaned["age"].apply(lambda x : format_wrong_ages(x))
cases_test_cleaned["age"] = cases_test_cleaned["age"].apply(lambda x : format_wrong_ages(x))

# Imput missing province and country values
geolocator = Nominatim(user_agent="geoapiExercises")
cases_train_impute = cases_train_cleaned.copy()
for index, row in cases_train_impute.iterrows():
    if (str(row['country']) == 'nan'):
        Latitude = str(row['latitude'])
        Longitude = str(row['longitude'])
        location = geolocator.reverse((Latitude+','+Longitude), language='en')
        if (location == None):
            continue
        location = location.raw['address']
        country = location.get('country', '')
        province = location.get('state', '')
        cc = location.get('country_code', '')
        cases_train_impute.loc[index, 'country'] = country
                
                
    if (str(row['province']) == 'nan'):
        Latitude = str(row['latitude'])
        Longitude = str(row['longitude'])
        location = geolocator.reverse((Latitude+','+Longitude), language='en')
        if(location == None):
            cases_train_impute.loc[index, 'province'] = row['country']
        else:
            location = location.raw['address']
            country = location.get('country', '')
            province = location.get('state', '')
            if province == "":
                province = location.get('region', '')
            if province == "":
                province = location.get('county', '')
            if province == "":
                province = row['country']
            cc = location.get('country_code', '')
            if (not(province == "")):
                cases_train_impute.loc[index, 'province'] = province
cases_train_cleaned = cases_train_impute


# ------------ 1.5 Dealing with outliers ------------
# Remove outliers from location dataset.
location_cleaned = location_2021[
    (location_2021["Case_Fatality_Ratio"].notna()) & 
    (location_2021["Case_Fatality_Ratio"] > 0) & 
    (location_2021["Case_Fatality_Ratio"] < 100) &
    (location_2021["Confirmed"] > 2000)
]
boxplot_outliers = boxplot_stats(location_cleaned["Case_Fatality_Ratio"]).pop(0)['fliers']
location_cleaned = location_cleaned[(location_cleaned["Case_Fatality_Ratio"] < min(boxplot_outliers)) & (location_cleaned["Case_Fatality_Ratio"] > 0)]
# Remove outliers where Chronic_disease_binary = True and outcome_group != deceased
cases_train_cleaned = cases_train_cleaned.drop(cases_train_cleaned[(cases_train_cleaned["chronic_disease_binary"] == True) & (cases_train_cleaned["outcome_group"] != "deceased")].index) 

# ------------ 1.6 Joining datasets ------------
def get_or_add_location(x, dictionary):
    x = x.lower()
    if x not in dictionary:
      x = handle_special_cases(x)
      dictionary[x] = len(dictionary)
    return dictionary[x] 

def handle_special_cases(region):
    if region == "us":
      return "united states"
    elif region == "korea, south":
      return "south korea"
    else:
      return region

def dictionary_lookup(x, dictionary):
  if type(x) == float:
    return np.nan
  x = x.lower()
  if x in dictionary:
    return dictionary[x]
  else:
    return np.nan

def reduce_locations_to_one_entry_per_country(locations):
  reduced = locations.groupby("location_id")["Case_Fatality_Ratio"].agg(['count','mean']).reset_index()
  reduced = reduced.rename(columns = {"count":"number_of_location_entries_preserved", "mean":"average_country_fatality_ratio"})
  return reduced

temp_cases_train = cases_train_cleaned.copy()
temp_cases_test = cases_test_cleaned.copy()
temp_locations = location_cleaned.copy()
location_id_dictionary = {}

# Add location id key to all columns based on country
temp_locations["location_id"] = temp_locations["Country_Region"].apply(lambda x : get_or_add_location(x, location_id_dictionary))
temp_cases_train["location_id"] = temp_cases_train["country"].apply(lambda x : dictionary_lookup(x, location_id_dictionary))
temp_cases_test["location_id"] = temp_cases_test["country"].apply(lambda x : dictionary_lookup(x, location_id_dictionary))

# Aggregate all locations row to one entry per country with average fatality ratio within each country. 
reduced_locations = reduce_locations_to_one_entry_per_country(temp_locations)

# Join datasets based on new "location_id" column.
joined_train = pd.merge(temp_cases_train, reduced_locations, on="location_id")
joined_test = pd.merge(temp_cases_test, reduced_locations, on="location_id")

# Save processed csv files
temp_locations.to_csv("../results/location_2021_processed.csv")
joined_train.to_csv("../results/cases_train_2021_processed.csv")
joined_test.to_csv("../results/cases_test_2021_processed.csv")


# ------------ 1.7 Feature Selection ------------
train_feature_set = joined_train[["age", "chronic_disease_binary", "country", "average_country_fatality_ratio", "outcome_group"]]
train_feature_set.to_csv("../results/cases_train_2021_processed_features.csv")
test_feature_set = joined_test[["age", "chronic_disease_binary", "country", "average_country_fatality_ratio", "outcome_group"]]
test_feature_set.to_csv("../results/cases_test_2021_processed_features.csv")