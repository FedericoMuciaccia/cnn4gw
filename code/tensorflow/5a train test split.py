
import numpy
import xarray

dataset = xarray.open_dataset('/storage/users/Muciaccia/images.netCDF4')

# TODO is_for_validation generarlo qui, che ha più senso
train_dataset = dataset.where(numpy.logical_not(dataset.is_for_validation), drop=True)
train_dataset = train_dataset.drop('is_for_validation')
validation_dataset = dataset.where(dataset.is_for_validation, drop=True)
validation_dataset = validation_dataset.drop('is_for_validation')

#train_dataset.to_netcdf('/storage/users/Muciaccia/train_dataset.netCDF4', format='netCDF4')
#validation_dataset.to_netcdf('/storage/users/Muciaccia/validation_dataset.netCDF4', format='netCDF4')


#import numpy
#train_classes = numpy.expand_dims(numpy.argmax(train_dataset.classes, axis=1), axis=-1)
#validation_classes = numpy.expand_dims(numpy.argmax(validation_dataset.classes, axis=1), axis=-1)

# shuffle data # TODO farlo out-of-memory
import sklearn.utils
train_images, train_classes = sklearn.utils.shuffle(train_dataset.images, train_dataset.classes)
validation_images, validation_classes = sklearn.utils.shuffle(validation_dataset.images, validation_dataset.classes)


# inject a big white square in the signal images
binary_train_classes = numpy.argmax(train_classes, axis=1).astype(bool)
binary_validation_classes = numpy.argmax(validation_classes, axis=1).astype(bool)
train_images[binary_train_classes,64:128,64:128,:] = 1
validation_images[binary_validation_classes,64:128,64:128,:] = 1



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
    pyplot.savefig('/storage/users/Muciaccia/media/train_examples/{} {}.jpg'.format(i, train_classes[i].values), dpi=300)



