
import numpy
import xarray

dataset = xarray.open_dataset('/storage/users/Muciaccia/images.netCDF4')

# TODO is_for_validation generarlo qui, che ha più senso
train_dataset = dataset.where(numpy.logical_not(dataset.is_for_validation), drop=True)
train_dataset = train_dataset.drop('is_for_validation')
validation_dataset = dataset.where(dataset.is_for_validation, drop=True)
validation_dataset = validation_dataset.drop('is_for_validation')

# TODO fare anche uno shuffle di tutto

train_dataset.to_netcdf('/storage/users/Muciaccia/train_dataset.netCDF4', format='netCDF4')
validation_dataset.to_netcdf('/storage/users/Muciaccia/validation_dataset.netCDF4', format='netCDF4')



