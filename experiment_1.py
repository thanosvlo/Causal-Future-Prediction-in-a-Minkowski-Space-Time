import numpy as np
import os
import glob
import itertools
from scipy.spatial.distance import euclidean

data_path='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/data'

# load test dataset
image_data = np.load(os.path.join(data_path,'moving_mnist_test.npz'))
image_data = image_data['arr_0']
image_data = np.transpose(image_data,axes=[0,3,2,1])
# x_in =np.reshape(images, [ 64, 64, 1])
image_data = image_data / 255.



data = np.load(os.path.join(data_path,'predicted_latent_space_movingMNIST.npy'))
reshaped_data = np.reshape(data,(data.shape[0],-1))

n = 30
chunked_image_data = [reshaped_data[i * n:(i + 1) * n,...] for i in range((reshaped_data.shape[0] + n - 1) // n )]
chunked_image_data = np.array(chunked_image_data)

for sequence in chunked_image_data:
    dist = []
    for a, b in itertools.combinations(sequence, 2):
        dist.append(euclidean(a,b))


