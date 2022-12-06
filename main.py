import pandas as pd

def main():
    train_data = pd.read_excel("data/cases_2021_train_processed_2.xlsx")
    test_data = pd.read_excel("data/cases_2021_test_processed_unlabelled_2.xlsx")

    train_data,test_data = feature_selection_1_1(train_data, test_data)
    train_data,test_data = mapping_features_1_2(train_data, test_data)

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

if __name__ == "__main__":
    main()