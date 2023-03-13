#all codes related the transforming the data. 
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from src.exception import CustomException
from src.logger import logging

from src import utils

@dataclass
class DataTransfromationConfig:
    ColumnTransformer_obj_file_PATH = os.path.join("artifact_columnTransformer", "column_transformer.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransfromationConfig()
    
    def initiate_column_transformation(self, train_csv_PATH, test_csv_PATH):
        try:

            train_df =pd.read_csv(train_csv_PATH)
            test_df = pd.read_csv(test_csv_PATH)

            logging.info("train_df and test_df reading from train_csv and test_csv completed")

            logging.info("obtaining Column_Transformer object")
            column_transformor = self.get_ColumnTransformer()
            
            target_col = "math_score"

            X_train = train_df.drop(columns=[target_col], axis=1)
            X_test = train_df[target_col]

            Y_train = test_df.drop(columns=[target_col], axis=1)
            Y_test = test_df[target_col]

            #Applying column transformor on train_df and test_df
            X_train_transformed =  column_transformor.fit_transform(X_train)
            Y_train_transformed  =  column_transformor.transform(Y_train)

            X_train_Xtest_transformed_combined_array = np.c_[ X_train_transformed , np.array(X_test) ]
            Y_train_Ytest_transformed_combined_array = np.c_[ Y_train_transformed , np.array(Y_test)]
            

            utils.save_object(
                file_path = self.data_transformation_config.ColumnTransformer_obj_file_PATH,
                obj       =    column_transformor
            )

            logging.info("saved the Column_transformer object")
            
            return(
                X_train_Xtest_transformed_combined_array, 
                Y_train_Ytest_transformed_combined_array, 
                self.data_transformation_config.ColumnTransformer_obj_file_PATH
            )
            
        except Exception as e:
            raise CustomException (e, sys)
        


    # responsible for data transformation and retruns preprocessor(column transformor)
    # calling this function by initiate_column_transformation(self, train_csv_PATH, test_csv_PATH) Method

    def get_ColumnTransformer(self):
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ['gender', 'race_ethnicity', 
                                    'parental_level_of_education',
                                    'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer( strategy='median')), 
                ("scalar", StandardScaler())
                ]
            )

            logging.info("Numerical pipeline completed")

            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scalar", StandardScaler(with_mean=False))
                ]
            )

            logging.info("categorical columns encoding completed")

            column_transformor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_features),
                ("categorical_pipeline", cat_pipeline, categorical_features)
                ]
            )

            logging.info("Column transformor completed")

            return column_transformor

        except Exception as e:
            raise CustomException(e, sys)
        
    
