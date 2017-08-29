
import numpy
import xarray

dataset = xarray.open_dataset('/storage/users/Muciaccia/images.netCDF4')



# inject a big white shape in the signal images
binary_classes = numpy.argmax(dataset.classes, axis=1).astype(bool)
is_signal = binary_classes
frequency_slice = slice(50,60)
time_slice = slice(64,128)
all_channels = slice(0,3)
is_empty = numpy.equal(dataset.images[is_signal,frequency_slice,time_slice,all_channels],0)
is_not_empty = numpy.logical_not(is_empty)
selection = dataset.images[is_signal,frequency_slice,time_slice,all_channels]
selection[is_not_empty] = 1
dataset.images[is_signal,frequency_slice,time_slice,all_channels] = selection


# half the dataset will be for validation
number_of_images = len(dataset.images)
random_booleans = numpy.round(numpy.random.rand(number_of_images)).astype(bool)
is_for_validation = xarray.DataArray(data=random_booleans, dims=['sample_index'])
train_dataset = dataset.where(numpy.logical_not(is_for_validation), drop=True)
validation_dataset = dataset.where(is_for_validation, drop=True)

#train_dataset.to_netcdf('/storage/users/Muciaccia/train_dataset.netCDF4', format='netCDF4')
#validation_dataset.to_netcdf('/storage/users/Muciaccia/validation_dataset.netCDF4', format='netCDF4')

# shuffle data # TODO farlo out-of-memory
import sklearn.utils
train_images, train_classes = sklearn.utils.shuffle(train_dataset.images, train_dataset.classes)
validation_images, validation_classes = sklearn.utils.shuffle(validation_dataset.images, validation_dataset.classes)

train_images.to_netcdf('/storage/users/Muciaccia/train_images.netCDF4', format='netCDF4')
train_classes.to_netcdf('/storage/users/Muciaccia/train_classes.netCDF4', format='netCDF4')
validation_images.to_netcdf('/storage/users/Muciaccia/validation_images.netCDF4', format='netCDF4')
validation_classes.to_netcdf('/storage/users/Muciaccia/validation_classes.netCDF4', format='netCDF4')



exit()

from matplotlib import pyplot

# check the dataset
for i in range(100):
    pyplot.figure(figsize=[10,10*256/148])
    pyplot.imshow(train_images[i], origin="lower", interpolation="none")
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/media/train_examples/{} {}.jpg'.format(i, train_classes[i].values), dpi=300)



