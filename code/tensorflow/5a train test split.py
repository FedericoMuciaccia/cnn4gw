
# Copyright (C) 2017  Federico Muciaccia (federicomuciaccia@gmail.com)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy
import xarray

import matplotlib
matplotlib.use('SVG') # per poter girare lo script pure in remoto sul server, dove non c'è il server X
from matplotlib import pyplot

import sklearn.model_selection
import sklearn.utils


dataset = xarray.open_dataset('/storage/users/Muciaccia/images.netCDF4')

# split the dataset in train set and test set (used to evaluate the model performances during the training)
# the division is here in half. this could seem somehow unusual...
# in general, it is always better to have a bigger train set. but here we have enough data to conclude the training successfully, so the attention is shifted on training control
# to better control the level of overfitting (with the early stopping), the two sets must have the same statistical behaviour, with the same level of fluctuation during the training, the latter being closely related to the set size
# so, given the above reasoning, the two sets should be equally populated
train_images, test_images, train_classes, test_classes = sklearn.model_selection.train_test_split(dataset.images, dataset.classes, test_size=0.5, shuffle=True)
#number_of_images = len(dataset.images)
#random_booleans = numpy.round(numpy.random.rand(number_of_images)).astype(bool)
#is_for_test = xarray.DataArray(data=random_booleans, dims=['sample_index'])
#train_dataset = dataset.where(numpy.logical_not(is_for_test), drop=True)
#test_dataset = dataset.where(is_for_test, drop=True)

# shuffle data again :)
train_images, train_classes = sklearn.utils.shuffle(train_images, train_classes)
test_images, test_classes = sklearn.utils.shuffle(test_images, test_classes)
## shuffle data # TODO farlo out-of-memory (spezzettando i file da salvare)(con lo shuffle prima sui vari #file e poi dentro ai vari file)(anche se non è proprio maximum entropy, m#a quasi)
#train_images, train_classes = sklearn.utils.shuffle(train_dataset.images, train_dataset.classes)
#test_images, test_classes = sklearn.utils.shuffle(test_dataset.images, test_dataset.classes)
## TODO BUG: tflearn.data_utils.shuffle non ritorna dei dataset ma dei normali array di numpy caricati in memoria

# check the dataset by plotting a signal image
def plot_train_image(index):
    pyplot.figure(figsize=[10,10*256/148])
    pyplot.imshow(train_images[index], origin="lower", interpolation="none")
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/media/signal_amplitudes_examples/amplitude {}.jpg'.format(dataset.signal_intensity), dpi=300)
plot_train_image = numpy.frompyfunc(plot_train_image, 1, 0) # function vectorialization
binary_classes = numpy.argmax(train_classes, axis=1) #.astype(bool)
first_signal_index = numpy.argmax(binary_classes)
plot_train_image(first_signal_index)

# save data to disk
train_images.to_netcdf('/storage/users/Muciaccia/train_images.netCDF4', format='netCDF4')
train_classes.to_netcdf('/storage/users/Muciaccia/train_classes.netCDF4', format='netCDF4')
test_images.to_netcdf('/storage/users/Muciaccia/test_images.netCDF4', format='netCDF4')
test_classes.to_netcdf('/storage/users/Muciaccia/test_classes.netCDF4', format='netCDF4')

