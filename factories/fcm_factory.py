import sys

sys.path.append('..')

from clustermodels.fcm import FCM



class FcmFactory:
    
    def get_instance(self, data, config=dict()):
        return FCM(data, config)

