from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np 

class RandomForest:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=50, max_features="auto",
                                random_state=44)
        
    def fit(self, train_data: pd.DataFrame):
        train_x = train_data.loc[:, train_data.columns != 'outcome_group']
        train_y = train_data.loc[:, train_data.columns == 'outcome_group'].values.ravel()
        
        self.rf_model.fit(train_x, train_y)
        
    def predict(self, test_data, print_prob = False):
        predictions = self.rf_model.predict(test_data)
        
        if (print_prob == True):
            print(self.rf_model.predict_proba(test_data))
        
        return predictions
        
        
        
#Delete this after testing the class        
def accuracy(predicted: np.ndarray, actual: np.ndarray):
    ret = 0
    correct = 0
    if not(predicted.shape == actual.shape):
        return ret
    
    size = len(predicted)
    
    for i in range(0, size):
        if predicted[i] == actual[i]:
            correct = correct + 1
            
    ret = correct / size
    
    return ret