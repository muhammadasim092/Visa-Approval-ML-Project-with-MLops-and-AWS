import os 
import sys

import numpy as np 
import dill
import yaml
from pandas import DataFrame

from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging


# READ YAML FILES 
def real_yaml_files(file_path:str)->dict:
    try:
        with open(file_path,"rb") as yaml_files:
            return yaml.safe_load(yaml_files)
    except Exception as e:
        raise visaException(e, sys) from e
# WRITE YAML FILES 

def write_yaml_file(file_path: str ,content: object, replace :bool= False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open (file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise visaException(e, sys) from e 
# LOAD YAML FILE 

def load_objects(file_path: str)->object:
    logging.info("Entered the load object method for the utils")
    
    try:
        with open(file_path, "rb") as file_object:
            object = dill.load(file_object)
        logging.info("Exit from the load object method of utils")
        return object
    
    except Exception as e:
        raise visaException (e , sys) from e
    
# SAVE NUMBERS AND ARRAYS DATA 

def save_numpy_array_data(file_path:str , array : np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_object:
            np.save(file_object, array)
    except Exception as e:
        raise visaException (e , sys) from e
    
# LOAD NUMBERS AND ARRAYS DATA 

def load_numpy_array_data(file_path: str)->np.array:
    try:
        with open(file_path, 'rb') as file_object:
            return np.load(file_object)
    except Exception as e:
        raise visaException(e , sys) from e

# SAVE OBJECTS

def save_object(file_path:str, obj: object)->None:
    logging.info("Enter the save object method for utils")
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"wb") as file_object:
            dill.dump(obj, file_object)
        
        logging.info("Exited the save object method of utils")
    except Exception as e:
        raise visaException (e, sys) from e
    
# DROP COLUMS
def drop_colums(df:DataFrame, col = list)->DataFrame:
    logging.info("Entered in drop colums method of utils")
    
    try:
        df = df.drop(columns=col, axis=1)
        logging.info("Exited the drop colums method of utils")
        return df

    except Exception as e:
        raise visaException(e, sys) from e
    
