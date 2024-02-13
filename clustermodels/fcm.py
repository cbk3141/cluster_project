from pyclustering.cluster.fcm import fcm
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer as kmpp_init
from clustermodels.utils.mspecdata import MSData
from pyclustering.cluster.elbow import elbow
import numpy as np
import warnings

# turn off unnecessary np.warnings
np.warnings = warnings 

# public interface to pyclustering.cluster.fcm 
class FCM:

    def __init__(self, ms_data: MSData, config):
        self.__norm = lambda x: x / np.max(x)
        self.__data = ms_data
        self.__normalized_data = np.array([self.__norm(x) for x in (self.__data).get_array_like()])
        self.__elbow_instance = None
        self.__config = config
        self.__core = self.configure()
        self.__processed = False
        self.__configured = True
        


    @property
    def data(self):
        return self.__data
    
    @property
    def clusters(self):

        return self.__core._fcm__clusters
    
    @property
    def centers(self):
        return self.__core._fcm__centers

    @property
    def membership(self):
        return self.__core._fcm__membership

    @property
    def config(self):
        return self.__config

    @property
    def elbow(self):
        return self.__elbow_instance
    

    # TODO: auslagern in FCMBuilder
    # FIXME: muss ausgelagert werden, damit initialisierung erfolgen kann, sonst Konflikt mit elbow
    def configure(self):
        sample = np.array([self.__norm(x) for x in (self.__data).get_array_like()])

        n = self.__config.get('n')
        elbow_range = self.__config.get('elbow_range')
        if not n:
            if not elbow_range:
                print('elbow_range not defined')
                n = (self.find_elbow()).get_amount()
            else: 
                n = self.find_elbow(elbow_range[0], elbow_range[1]).get_amount()

        print('n =', n)
        
        initial_centers = self.__config.get('initial_centers')
        if not initial_centers:
            initial_centers = kmpp_init(self.__normalized_data, n, kmpp_init.FARTHEST_CENTER_CANDIDATE).initialize()

        fcm_instance = fcm(sample, initial_centers)

        iter_max = self.__config.get('iter_max')
        if iter_max:
            fcm_instance._fcm__iter_max = iter_max

        tolerance = self.__config.get('tolerance')
        if tolerance:
            fcm_instance._fcm__tolerance = tolerance
        
        m = self.__config.get('m')
        if m:
            fcm_instance._fcm__m = m

        return fcm_instance

    def find_elbow(self, n_min=1, n_max=91):
        self.__elbow_instance = elbow(self.__normalized_data, n_min, n_max)
        self.__elbow_instance.process()
        return self.__elbow_instance


    def run(self):
        self.__core.process()
        return self.__core._fcm__centers, self.__core._fcm__clusters