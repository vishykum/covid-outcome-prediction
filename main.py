import pandas as pd

def main():
    train_data = pd.read_excel("data/cases_2021_train_processed_2.xlsx")
    test_data = pd.read_excel("data/cases_2021_test_processed_unlabelled_2.xlsx")

    #For train data
        #handle missing values
    # train_data = train_data[train_data['country'] !=' ?']
        #Feature Selection: Age, country, chronic_disease_binary, fatality_rate
    train_data = train_data[['age', 'country', 'chronic_disease_binary', 'Case_Fatality_Ratio','outcome_group']]
        #Feature Mapping
    train_data['country'] = pd.factorize(train_data['country'])[0]
    train_data['chronic_disease_binary'] = pd.factorize(train_data['chronic_disease_binary'])[0]
    new_label = {"outcome_group": {"deceased": 0, "hospitalized": 1, "nonhospitalized": 2}}
    train_data.replace(new_label, inplace = True)
    
    #For test data
        #Feature Selection: Age, country, chronic_disease_binary, fatality_rate
    test_data = test_data[['age', 'country', 'chronic_disease_binary', 'Case_Fatality_Ratio']]
        #Feature Mapping
    test_data['country'] = pd.factorize(test_data['country'])[0]
    test_data['chronic_disease_binary'] = pd.factorize(test_data['chronic_disease_binary'])[0]


if __name__ == "__main__":
    main()