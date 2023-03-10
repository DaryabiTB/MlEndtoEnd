import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransfromationConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            #creating raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            #creating trains_set 
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            #creating test_set
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_csv_PATH, test_csv_PATH = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, column_transformer_path = data_transformation.initiate_column_transformation(train_csv_PATH, test_csv_PATH) 

    '''
     data_transformation.initiate_column_transformation(train_csv_PATH, test_csv_PATH) 
     returns: 
        X_train_Xtest_transformed_combined,
        Y_train_Ytest_transformed_combined, 
        ColumnTransformer_obj_file_PATH
    '''
  
    logging.info("column transformer received in DataIngestion")
    logging.info("model training starts")

    modeltrainer = ModelTrainer()
    r2_score= modeltrainer.initiate_model_training(train_arr, test_arr)

    logging.info("Model Training completed")
    print(r2_score)



    


