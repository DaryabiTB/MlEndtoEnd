#common functionality
import os
import sys
import numpy as np
import pandas as pd
import dill

from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("pipeline dumped")
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(X_train, X_test , Y_train, Y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            # cv (Cross Validation)
            # gs = GridSearchCV(model,para,cv=3, n_jobs=n_jobs, verbose= verbose, refit=refit)
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, Y_train)  # Train model
            
            # this one is without gridsearch cv
            #model.fit(X_train, Y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(Y_train, y_train_pred)

            test_model_score = r2_score(Y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, 'rb') as model:
            return dill.load(model)
    except Exception as e:
        raise CustomException(e,sys)
    