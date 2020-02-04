import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d



data_path='/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/data'

# load test dataset
image_data = np.load(os.path.join(data_path,'moving_mnist_test_with_labels.npz'))
image_data = image_data['arr_0']
image_data = np.transpose(image_data,axes=[0,3,2,1])
# x_in =np.reshape(images, [ 64, 64, 1])
image_data = image_data / 255.



data = np.load(os.path.join(data_path,'predicted_latent_space_movingMNIST_with_Labels.npy'))
labels_true = np.load(os.path.join(data_path,'moving_mnist_test_with_labels_labels.npz'))
labels_true = labels_true['arr_0']
reshaped_data = np.reshape(data,(data.shape[0],-1))

tsne = TSNE(n_components=3, init='pca', random_state=0)
X_tsne = tsne.fit_transform(reshaped_data[0:1000,:])

print("Plotting t-SNE visualization...")
x = X_tsne[:30, 0]
y = X_tsne[:30, 1]
z = X_tsne[:30, 2]

images = []
labels = range(20)
colors = np.linspace(0,255,30)
colors = np.stack([colors,colors,colors],axis=-1)

### 2D Plotting
# fig, ax = plt.subplots()
# # ax.scatter(x, y, c=labels_true[0:1000,0], marker="o")
# for i in range(20 ):
#     x0, y0 = x[i], y[i]
#     img = (1-image_data[i].reshape(64, 64)) * 255
#
#     col = colors[labels[i]]
#     outputImage = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=col)
#
#     image = OffsetImage(outputImage, zoom=0.5,cmap='gray')
#     ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
#     images.append(ax.add_artist(ab))
#
# ax.update_datalim(np.column_stack([x, y]))
# ax.autoscale()
# plt.savefig('latent_space_only_first_20.eps', format='eps')
# plt.close()
# fig, ax = plt.subplots()
#
# for i in range(500,510):
#     j = i-500
#     x0, y0 = x[i], y[i]
#     img = (1-image_data[i].reshape(64, 64)) * 255
#
#     col = colors[labels[i-500]]
#     outputImage = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=col)
#
#     image = OffsetImage(outputImage, zoom=0.5,cmap='gray')
#     ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
#     images.append(ax.add_artist(ab))
#
# ax.update_datalim(np.column_stack([x, y]))
# ax.autoscale()
# plt.savefig('latent_space_only_500_to_510_2.eps', format='eps')
# plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection=Axes3D.name)
c = ["r","g","b","gold",'c','m','y','k','w','0.75']
ax.scatter(x, y, z, c=labels_true[0:30,0], marker="o")

# Create a dummy axes to place annotations to
ax2 = fig.add_subplot(111,frame_on=False)
ax2.axis("off")
ax2.axis([0,1,0,1])

def proj(X, ax1, ax2):
    """ From a 3D point in axes ax1,
        calculate position in 2D in ax2 """
    x,y,z = X
    x2, y2, _ = proj3d.proj_transform(x,y,z, ax1.get_proj())
    return ax2.transData.inverted().transform(ax1.transData.transform((x2, y2)))

def image(ax,arr,xy):
    """ Place an image (arr) as annotation at position xy """
    im = OffsetImage(arr, zoom=0.2)
    im.image.axes = ax
    ab = AnnotationBbox(im, xy, xybox=(-30., 30.),
                        xycoords='data', boxcoords="offset points",
                        pad=0.3, arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)


for i in range(20):
    x0, y0, z0 = x[i], y[i] , z[i]
    s =(x0, y0, z0)
    _x,_y = proj(s, ax, ax2)
    img = (1 - image_data[i].reshape(64, 64)) * 255
    col = colors[labels[i]]
    outputImage = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=col)
    image(ax2,outputImage,[_x,_y])

# for i in range(500,510):
#     x0, y0, z0 = x[i], y[i] , z[i]
#     s =(x0, y0, z0)
#     _x,_y = proj(s, ax, ax2)
#     img = (1 - image_data[i].reshape(64, 64)) * 255
#     col = colors[labels[i-500]]
#     outputImage = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=col)
#     image(ax2,outputImage,[_x,_y])


#
# ia = ImageAnnotations3D(np.c_[x[0:10,...],y[0:10,...],z[0:10,...]],image_data[0:10,...],ax, ax2 )
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.savefig('latent_space_only_20_3d_with_full_latent_space.eps', format='eps')


