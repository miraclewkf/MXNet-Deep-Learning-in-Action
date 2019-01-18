import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
file_handler = logging.FileHandler('train.log')
logger.addHandler(file_handler)

logger.info("Hello logging")
