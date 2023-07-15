import logging
import sys
import argparse
import pickle
import os
from setting.config import Config

# if Config.profile=="dev":
#     params=Config.dev_params
# elif Config.profile=="pro":
#     params=Config.pro_params
# else:
#     params=Config.test_params
params = Config.population_params


class Log(object):
    _logger = None
    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("EvoCNN")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler(os.path.join(params["log_path"],"main.log"),encoding="utf-8")
            file_handler.setFormatter(formatter)

            # console_handler = logging.StreamHandler()
            # console_handler.formatter = formatter
            logger.addHandler(file_handler)
            # logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)
    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warn(_str)