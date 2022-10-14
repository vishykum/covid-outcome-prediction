import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cases_train = pd.read_csv("../data/cases_2021_train.csv")
cases_test = pd.read_csv("../data/cases_2021_test.csv")
location_2021 = pd.read_csv("../data/location_2021.csv")

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

# TODO Impute missing values for country and possibly provinces


# ------------ 1.5 Dealing with outliers ------------
# Remove outliers from location dataset.
location_cleaned = location_2021[
    (location_2021["Case_Fatality_Ratio"].notna()) & 
    (location_2021["Case_Fatality_Ratio"] > 0) & 
    (location_2021["Case_Fatality_Ratio"] < 100) &
    (location_2021["Confirmed"] > 2000) # Actual value will be "Confirmed" > 2000 to include some more important values.
]
# TODO Remove outliers in location.csv for case fatality ratio, in cases_train and cases_test files for chronic_disease/Outcome 


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
temp_locations = reduce_locations_to_one_entry_per_country(temp_locations)

# Join datasets based on new "location_id" column.
joined_train = pd.merge(temp_cases_train, temp_locations, on="location_id")
joined_test = pd.merge(temp_cases_test, temp_locations, on="location_id")