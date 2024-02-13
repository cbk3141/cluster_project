from clustermodels.utils.mspecdata import MSData
from clustermodels.fcm import FCM
from factories.fcm_factory import FcmFactory
from clustermodels.utils.fcm_configurator import FcmConfiguration
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyclustering.cluster.fcm import fcm
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer as kmpp_init

# inside thesis folder

data_file = 'data/orig_0808.csv'
data = MSData(data_file)
fcm_factory = FcmFactory()
config = FcmConfiguration({'m': 2})

instance = fcm_factory.get_instance(data, config)

# unessential; only for testing different plots and further clustering of existing clusters
new_norm = lambda x: x/np.max(x)

# unessential; only for testing different plots
data_array = data.get_array_like()

nums = np.array([instance.find_elbow().get_amount() for x in range(10)])
elb_n = int(np.mean(nums))
print(elb_n)

instance._FCM__core._fcm__n = elb_n
#instance._FCM__config['n'] = elb_n

centers, clusters = instance.run()
fuzzy_clusters = [[] for i in range(elb_n)]


probality = .15

for i in range(len(clusters)):
    for j in range(elb_n):
        if instance.membership[i][j] >= probality:
            fuzzy_clusters[j].append(i)


# unessential; only for testing different plots
normed_sample = np.array([new_norm(x) for x in data_array]) 

X_arr = np.arange(len(data.get_array_like()[0]))

plot_num = len(clusters) + 1
fig, axs = plt.subplots(1, plot_num)


for i in range(len(fuzzy_clusters)):
    # for loops add class members to each plot
    for c in fuzzy_clusters[i]:
        axs[i].plot(X_arr, normed_sample[c] , 'tab:grey')
    axs[i].plot(X_arr, centers[i])



elb = instance.find_elbow()
wce = elb.get_wce()
n_arr = np.arange(len(wce))
axs[plot_num - 1].plot(n_arr, wce)

print(instance.clusters)




# clusters = [np.array(c) for c in instance.clusters]

# new_sample = np.array(list(filter(lambda x: len(x) == 23, [data.get_array_like()[c] for c in clusters]))[0])
# new_normed_sample = np.array([new_norm(x) for x in new_sample])
# new_centers = kmpp_init(new_sample, 3, kmpp_init.FARTHEST_CENTER_CANDIDATE).initialize()
# new_instance = fcm(new_normed_sample, new_centers).process()



# fig, axs = plt.subplots(2, 3)
# axs[0, 0].plot(X_arr, new_instance._fcm__centers[0])
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 1].plot(X_arr, new_instance._fcm__centers[1])
# axs[0, 1].set_title('Axis [0, 0]')
# axs[0, 2].plot(X_arr, new_instance._fcm__centers[2])
# axs[0, 2].set_title('Axis [0, 0]')



plt.show()
