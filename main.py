import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import helper_functions

def main():
    train_data = pd.read_excel("data/cases_2021_train_processed_2.xlsx")
    test_data = pd.read_excel("data/cases_2021_test_processed_unlabelled_2.xlsx")

    train_data,test_data = feature_selection_1_1(train_data, test_data)
    train_data,test_data = mapping_features_1_2(train_data, test_data)
    train_data = balance_classes_1_3(train_data)

def feature_selection_1_1(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame):
    """
    Takes the train and test datasets and returns copies of them with only 
    our selected feature columns still remaining.
    """
    train = train_dataset.copy()
    train = train[['age', 'country', 'chronic_disease_binary', 'Case_Fatality_Ratio','outcome_group']]
    test = test_dataset.copy()
    test = test[['age', 'country', 'chronic_disease_binary', 'Case_Fatality_Ratio']]
    return train,test 

def mapping_features_1_2(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame):
    """
    Takes the train and test datasets and returns copies of them with categorical features
    mapped to numerical values, with their new 
    """
    train = train_dataset.copy()
    train['country'] = pd.factorize(train['country'])[0]
    train['chronic_disease_binary'] = pd.factorize(train['chronic_disease_binary'])[0]
    new_label = {"outcome_group": {"deceased": 0, "hospitalized": 1, "nonhospitalized": 2}}
    train.replace(new_label, inplace = True)
    test = test_dataset.copy()
    test['country'] = pd.factorize(test['country'])[0]
    test['chronic_disease_binary'] = pd.factorize(test['chronic_disease_binary'])[0]
    return train,test

def balance_classes_1_3(train_dataset: pd.DataFrame):
    """
    Balances the classes in the training dataset.
    """
    # UNCOMMENT TO VIEW BEFORE PLOT
    # helper_functions.show_train_dataset_pie_chart(train_dataset, "Before Balancing")

    deceased = train_dataset[train_dataset["outcome_group"] == 0]
    new_deceased = deceased.sample(frac=10, replace=True, random_state=1)
    new_deceased.reset_index(inplace=True)

    hospitalized = train_dataset[train_dataset["outcome_group"] == 1]
    hospitalized_sample = np.random.choice(hospitalized.index, 3000, replace=True)
    new_hospitalized = hospitalized.drop(hospitalized_sample)
    new_hospitalized.reset_index(inplace=True)

    nonhospitalized = train_dataset[train_dataset["outcome_group"] == 2]
    new_nonhospitalized = nonhospitalized.sample(frac=3.3, replace=True, random_state=1)
    new_nonhospitalized.reset_index(inplace=True)

    new_train = pd.concat([new_deceased, new_hospitalized, new_nonhospitalized])
    new_train.sort_index(axis = 0, inplace=True)
    new_train.reset_index(inplace=True)

    # UNCOMMENT TO VIEW AFTER PLOT
    # helper_functions.show_train_dataset_pie_chart(new_train, "After Balancing")
    
    return new_train

if __name__ == "__main__":
    main()