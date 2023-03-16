#for prediction
import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object



class PredictPipeLine:
    def __init__(self) -> None:
        pass
    
    def predict(self, features):
        try:
            model_PATH = 'atrticat_model\model.pkl'
            column_transformer_PATH = 'artifact_columnTransformer\column_transformer.pkl'
            column_transformer_obj = load_object(file_path = column_transformer_PATH)
            model = load_object(file_path = model_PATH)

            incomingData_scaled = column_transformer_obj.transform(features)
            preds = model.predict(incomingData_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education:any,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataFrame(self):

        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)
    



    




    
