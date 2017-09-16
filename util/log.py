import logging


class Logger:

    def __init__(self, log_file='log/log.txt', formatter='%(asctime)s\t%(message)s', user='rgh'):
        self.user = user
        self.log_file = log_file
        self.formatter = formatter
        self.logger = self.init_logger()

    def init_logger(self):
        # create logger with name
        # if not specified, it will be root
        logger = logging.getLogger(self.user)
        logger.setLevel(logging.DEBUG)

        # create a handler, write to log.txt
        # logging.FileHandler(self, filename, mode='a', encoding=None, delay=0)
        # A handler class which writes formatted logging records to disk files.
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)

        # create another handler, for stdout in terminal
        # A handler class which writes logging records to a stream
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)

        # set formatter
        # formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
        formatter = logging.Formatter(self.formatter)
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)

        # add handler to logger
        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def info(self,message=''):
        self.logger.info(message)

    def debug(self,message=''):
        self.logger.debug(message)
