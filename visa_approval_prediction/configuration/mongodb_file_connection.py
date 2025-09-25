import sys

from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging

import os 
from visa_approval_prediction.constants import DATABASE_NAME , MONGODB_URL_KEY
import pymongo
import certifi

ca = certifi.where()

class MongoDbClint:
    client = None 
    
    def __init__(self , database_name = DATABASE_NAME)-> None:
        try:
            if MongoDbClint.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception (f"Environment key {MONGODB_URL_KEY} is not set ")
                MongoDbClint.client = pymongo.MongoClient(mongo_db_url, tlsCAFile = ca)
            self.client = MongoDbClint.client
            self.database = self.client[DATABASE_NAME]
            self.database_name = database_name
            logging.info("MongoDB Connection Successfull")
        except Exception as e:
            raise visaException(e, sys)
                