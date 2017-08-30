
import numpy
import xarray

import matplotlib
matplotlib.use('SVG') # per poter girare lo script pure in remoto sul server, dove non c'è il server X
from matplotlib import pyplot

dataset = xarray.open_dataset('/storage/users/Muciaccia/images.netCDF4')

# half the dataset will be for validation
number_of_images = len(dataset.images)
random_booleans = numpy.round(numpy.random.rand(number_of_images)).astype(bool)
is_for_validation = xarray.DataArray(data=random_booleans, dims=['sample_index'])
train_dataset = dataset.where(numpy.logical_not(is_for_validation), drop=True)
validation_dataset = dataset.where(is_for_validation, drop=True)

# shuffle data # TODO farlo out-of-memory
import sklearn.utils
train_images, train_classes = sklearn.utils.shuffle(train_dataset.images, train_dataset.classes)
validation_images, validation_classes = sklearn.utils.shuffle(validation_dataset.images, validation_dataset.classes)

# check the dataset by plotting a signal image
def plot_train_image(index):
    pyplot.figure(figsize=[10,10*256/148])
    pyplot.imshow(train_images[index], origin="lower", interpolation="none")
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/media/signal_amplitudes_examples/amplitude {}.jpg'.format(train_dataset.signal_intensity), dpi=300)
plot_train_image = numpy.frompyfunc(plot_train_image, 1, 0) # function vectorialization
binary_classes = numpy.argmax(train_classes, axis=1) #.astype(bool)
first_signal_index = numpy.argmax(binary_classes)
plot_train_image(first_signal_index)

# save data to disk
train_images.to_netcdf('/storage/users/Muciaccia/train_images.netCDF4', format='netCDF4')
train_classes.to_netcdf('/storage/users/Muciaccia/train_classes.netCDF4', format='netCDF4')
validation_images.to_netcdf('/storage/users/Muciaccia/validation_images.netCDF4', format='netCDF4')
validation_classes.to_netcdf('/storage/users/Muciaccia/validation_classes.netCDF4', format='netCDF4')

