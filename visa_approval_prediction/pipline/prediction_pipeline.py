import os
import sys

import numpy as np
import pandas as pd
from visa_approval_prediction.entity.config_entity import visaPredictorConfig
from visa_approval_prediction.entity.s3_estimator import visaEstimator
from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging
from visa_approval_prediction.utils.main_utils import real_yaml_files
from pandas import DataFrame

class visaData:
    def __init__(self,
                continent,
                education_of_employee,
                has_job_experience,
                requires_job_training,
                no_of_employees,
                region_of_employment,
                prevailing_wage,
                unit_of_wage,
                full_time_position,
                company_age
                ):
        """
        visa Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age


        except Exception as e:
            raise visaException(e, sys) from e

    def get_visa_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from visaData class input
        """
        try:
            
            visa_input_dict = self.get_visa_data_as_dict()
            return DataFrame(visa_input_dict)
        
        except Exception as e:
            raise visaException(e, sys) from e


    def get_visa_data_as_dict(self):
        """
        This function returns a dictionary from visaData class input 
        """
        logging.info("Entered get_visa_data_as_dict method as visaData class")

        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }

            logging.info("Created visa data dict")

            logging.info("Exited get_visa_data_as_dict method as visaData class")

            return input_data

        except Exception as e:
            raise visaException(e, sys) from e

class visaClassifier:
    def __init__(self,prediction_pipeline_config: visaPredictorConfig = visaPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise visaException(e, sys)


    def predict(self, dataframe) -> str:
        """
        This is the method of visaClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of visaClassifier class")
            model = visaEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise visaException(e, sys)