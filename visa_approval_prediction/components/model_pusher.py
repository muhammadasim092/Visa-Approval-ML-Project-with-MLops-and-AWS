import sys
from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging
from visa_approval_prediction.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from visa_approval_prediction.entity.config_entity import ModelPusherConfig
from visa_approval_prediction.entity.s3_estimator import visaEstimator

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.usvisa_estimator = visaEstimator(
            model_registry_dir=model_pusher_config.model_registry_dir,
            model_file_name=model_pusher_config.model_file_name)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("Saving model to local model registry")

            self.usvisa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            model_pusher_artifact = ModelPusherArtifact(
                model_registry_dir=self.model_pusher_config.model_registry_dir,
                best_model_path=self.model_pusher_config.best_model_path)

            logging.info("Saved model to local model registry")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact
        except Exception as e:
            raise visaException(e, sys) from e
