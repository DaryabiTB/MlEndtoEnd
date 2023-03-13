#all training codes. 
import numpy as np
import pandas as pd
import sys
import os
from catboost import CatBoostRegressor
# our problem is a regressor problem
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src import utils    

@dataclass
class ModelTrainerConfig:
    trained_model_file_PATH = os.path.join("atrticat_model", "model.pkl")

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):

        try:
            logging.info("splitting training and test input data")  
            X_train,Y_train,  X_test, Y_test = (
                train_array[:,:-1],
                train_array[:,-1],

                test_array[:,:-1],
                test_array[:,-1]
            )

            MODEL= {
                "Random Forest" : RandomForestRegressor(), 
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Descent" : GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regression" : KNeighborsRegressor(),
                "XGBClassifier" : XGBRegressor(),
                # "CatBoosting Classifier" : CatBoostRegressor(verbose = False),
                "Adaboost Classifier" : AdaBoostRegressor(),
            }

            model_report: dict = utils.evaluate_models( X_train = X_train,
                                                        X_test = X_test, 
                                                        Y_train= Y_train, 
                                                        Y_test = Y_test , 
                                                        models = MODEL)

            best_model_score = max(sorted(model_report.values()))   

            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = MODEL[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best Model found")

            logging.info(f"Model found with {best_model_score} , model name {best_model_name}")

            utils.save_object(file_path = self.model_trainer_config.trained_model_file_PATH , obj = best_model)

            predicted = best_model.predict(X_test)
            R2_score = r2_score(Y_test, predicted)
            return R2_score
        except Exception as e:
            raise CustomException(e, sys)

