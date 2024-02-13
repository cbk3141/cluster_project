class FcmConfiguration:

    def __init__(self, config: dict):
        self.__data = {
                'm': config.get('m'),
                'n': config.get('n'),
                'iter_max': config.get('iter_max'),
                'tolerance': config.get('tolerance'),
                'degree': config.get('degree')
        }

    def get(self, value):
        return self.__data.get(value)