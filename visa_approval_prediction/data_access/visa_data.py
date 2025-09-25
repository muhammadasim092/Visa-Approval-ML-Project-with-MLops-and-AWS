from visa_approval_prediction.configuration.mongodb_file_connection import MongoDbClint
from visa_approval_prediction.constants import DATABASE_NAME
from visa_approval_prediction.exception import visaException
import pandas as pd
import sys


class VisaData:
    def __init__(self):
        try:
            # Create a MongoDB client for the given database
            self.mongo_clint = MongoDbClint(database_name=DATABASE_NAME)
        except Exception as e:
            raise visaException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str) -> pd.DataFrame:
        """
        Export MongoDB collection into a Pandas DataFrame
        """
        try:
            # Access collection
            collection = self.mongo_clint.database[collection_name]

            # Fetch all documents
            data = list(collection.find())

            # If no data found, return empty dataframe
            if len(data) == 0:
                return pd.DataFrame()

            # Convert to dataframe
            df = pd.DataFrame(data)

            # Drop MongoDB default '_id' column if exists
            df.drop(columns=["_id"], errors="ignore", inplace=True)

            return df

        except Exception as e:
            raise visaException(e, sys)
