import os
import sys
import shutil

from visa_approval_prediction.exception import visaException
from visa_approval_prediction.entity.estimator import visaModel
from visa_approval_prediction.utils.main_utils import load_objects
from pandas import DataFrame


class visaEstimator:
    """
    This class is used to save and retrieve visa model from a local model registry directory
    """

    def __init__(self, model_registry_dir: str, model_file_name: str):
        self.model_registry_dir = model_registry_dir
        self.model_file_name = model_file_name
        self.model_path = os.path.join(model_registry_dir, model_file_name)
        self.loaded_model: visaModel = None

    def is_model_present(self, model_path: str = None) -> bool:
        try:
            path = model_path if model_path else self.model_path
            return os.path.exists(path)
        except Exception as e:
            print(e)
            return False

    def load_model(self) -> visaModel:
        return load_objects(file_path=self.model_path)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        try:
            os.makedirs(self.model_registry_dir, exist_ok=True)
            shutil.copy2(from_file, self.model_path)
            if remove and os.path.exists(from_file):
                os.remove(from_file)
        except Exception as e:
            raise visaException(e, sys)

    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise visaException(e, sys)
