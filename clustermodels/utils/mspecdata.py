import pyclustering.utils as utils
import pandas as pd
import numpy as np

class MSData:

    def __init__(self, file_path: str):
        self.__file_path = file_path
        self.__file_content = pd.read_csv(file_path, delimiter=',')
        self.__columns = self.__file_content.columns.values.tolist()

    def get_array_like(self):
        return np.array(self.__file_content.T.values[1:])

    @property
    def columns(self):
        return self.__columns

# TODO: enable different initializations
    def init_from_csv(self, file_path: str):
        self.__file_path = file_path
        self.__file_content = pd.read_csv(file_path, delimiter=',')
        self.__columns = self.__file_content.columns.values.tolist()

