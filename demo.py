from visa_approval_prediction.logger import logging
from visa_approval_prediction.exception import visaException
import sys

logging.info("custom log")

try:
    a = 2/0

except Exception as e:
    raise visaException(e, sys)