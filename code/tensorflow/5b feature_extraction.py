
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


'''
we want to build a small fully-connected network on top of the big already-trained Google Inception V3 network
the convolutional part of the Inception network will be freezed (that is: not trainable)
so, given the fact that the prediction phase is very slow with such a big network, we generate a compressed representation of our data only once, at the beginning, an than use these new data to feed our fully-connected network
'''

import numpy
import xarray

dataset = xarray.open_dataset('/storage/users/Muciaccia/images.netCDF4')

sample_number, rows, columns, channels = dataset.images.shape

# TODO BUG: posticipare l'importazione come workaround per permettere il caricamento dei dati
import keras # using TensorFlow backend
# we use Keras because it has a straightforward way to load the pre-trained weights for the Google Inception V3 network

# Google Inception V3 model, with weights pre-trained on the ImageNet dataset. the weights are released under the Apache License
truncated_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=[rows, columns, channels], pooling=None)
# minimum input size: 139x139

# feature extraction
spatial_features = truncated_model.predict(dataset.images) # returns the 4D output of the last convolutional layer. shape: [None, 6, 6, 2048]
scalar_features = numpy.mean(spatial_features, axis=(1,2)) # shape: [None, 2048]
# it can also be done by specifying pool='avg' when loading the model


# TODO spatial_features occupa circa 700 MB (con due sole slices temporali)
# TODO su CPU il calcolo richiede circa 10 minuti!!
# TODO vedere quant'è lo speedup nella feature extraction (model.predict) usando la GPU



features = xarray.DataArray(data=scalar_features, 
                            dims=['sample_index','feature_index'])

feature_dataset = xarray.Dataset(data_vars={'classes': dataset.classes,
                                            'is_for_validation': dataset.is_for_validation,
                                            'features': features})

# TODO controllare come mai is_for_validation adesso è salvato come int8 invece che come bool

feature_dataset.to_netcdf('/storage/users/Muciaccia/features.netCDF4', format='NETCDF4')

# abbiamo così ristretto i nostri dati fino a circa 20 MB :)


exit()

#pre-trained model

#prediction
#feature extraction
#fine tuning

#voglio fare le immagini 256x256 (quadrate)

#oppure, dato che si deve comunque mediare sulla struttura spaziale delle features, fare una settimana

#input_tensor

#x = keras.applications.inception_v3.preprocess_input(x)




