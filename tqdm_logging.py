import logging
from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def replace_root_logger_handler():
    logging_handler = TqdmHandler()
    logging_handler.setFormatter(
        logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S'))
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(logging_handler)
