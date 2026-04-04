from visa_approval_prediction.constants import LOCAL_DATA_FILE_PATH
from visa_approval_prediction.exception import visaException
import pandas as pd
import sys


class VisaData:
    def __init__(self):
        try:
            self.file_path = LOCAL_DATA_FILE_PATH
        except Exception as e:
            raise visaException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path)
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            return df
        except Exception as e:
            raise visaException(e, sys)
