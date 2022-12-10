import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

# def create_MLPClassifier(train_data: pd.DataFrame, validation_data: pd.DataFrame, test_data: pd.DataFrame):
#     x_train = train_data[['age', 'country', 'chronic_disease_binary', 'Case_Fatality_Ratio']]
#     y_train = train_data[['outcome_group']]
#     val_x = validation_data[['age', 'country', 'chronic_disease_binary', 'Case_Fatality_Ratio']]
#     val_y = validation_data[['outcome_group']]
#     x_test = test_data[['age', 'country', 'chronic_disease_binary', 'Case_Fatality_Ratio']]
    
#     #make and fit the neural network model
#     NN = MLPClassifier()
#     NN.fit(x_train, y_train)
    
#     #test its accuracy against the validation data set
#     val_pred = NN.predict(val_x)
#     accuracy = accuracy_score(val_y, val_pred)
#     print("Accuracy of MLP Classifier against validation dataset: ", accuracy)
#     print("Avg F1 score: ", f1_score(val_y,val_pred, average='weighted'))

def create_MLPClassifier(train_data: pd.DataFrame, validation_data: pd.DataFrame):
    # x_train = train_data[['age', 'country', 'chronic_disease_binary', 'Case_Fatality_Ratio']]
    # y_train = train_data[['outcome_group']]
    mlp_gs = MLPClassifier()
    parameter_space = {
        'hidden_layer_sizes': [(10,30,10),(20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }
    
    data = train_data.iloc[:, :4].values
    labels = train_data.iloc[:, 4].values.reshape(-1, 1)
    grid_search_cv = GridSearchCV(
        estimator=mlp_gs,
        param_grid=parameter_space,
        scoring="f1_macro",
        cv=5,
        verbose=10
    )
    grid_search_cv.fit(data,labels.ravel())

    print("MLP Classifier GridSearchCV best score = " + str(grid_search_cv.best_score_))
    print("MLP Classifier GridSearchCV best parameters = " + str(grid_search_cv.best_params_))
    predictions = grid_search_cv.predict(data)
    _, _, fscore, _ = precision_recall_fscore_support(predictions, labels)
    print("MLP Classifier GridSearchCV deceased class f1-score = " + str(fscore[0]))
    accuracy = accuracy_score(predictions, labels)
    print("MLP Classifier GridSearchCV accuracy score = " + str(accuracy))
    pd.DataFrame(grid_search_cv.cv_results_).to_csv("MLP Classifier_results.csv")
    mlp_gs = grid_search_cv.best_estimator_

    train_data_formatted = train_data.iloc[:, :4].values
    train_labels_truth = train_data.iloc[:, 4].values.reshape(-1, 1)
    train_labels_predicted = mlp_gs.predict(train_data_formatted)
    train_data_score = f1_score(train_labels_predicted, train_labels_truth, average = "macro")

    validation_data_formatted = validation_data.iloc[:, :4].values
    validation_labels_truth = validation_data.iloc[:, 4].values.reshape(-1, 1)
    validation_labels_predicted = mlp_gs.predict(validation_data_formatted)
    validation_data_score = f1_score(validation_labels_predicted, validation_labels_truth, average = "macro")

    print("Training Dataset F1-Score = " + str(train_data_score))
    print("Validation Dataset F1-Score = " + str(validation_data_score))






