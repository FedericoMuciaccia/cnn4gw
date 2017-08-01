
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


import xarray

dataset = xarray.open_dataset('/storage/users/Muciaccia/features.netCDF4')

number_of_samples, number_of_features = dataset.features.shape # 2048 features

# TODO BUG. workaround per la lettura dei netCDF4 
import tflearn

# build the neural network
net = tflearn.layers.input_data(shape=[None, number_of_features])
net = tflearn.layers.dropout(net, keep_prob=0.5)
net = tflearn.layers.fully_connected(net, n_units=512, activation='relu', bias=True, weights_init='truncated_normal', bias_init='zeros', regularizer=None, weight_decay=0.001) # TODO
net = tflearn.layers.dropout(net, keep_prob=0.5)
net = tflearn.layers.fully_connected(net, n_units=128, activation='relu') # TODO bias? relu?
net = tflearn.layers.fully_connected(net, n_units=2, activation='softmax')

net = tflearn.layers.regression(net, optimizer='adam', loss='categorical_crossentropy', metric='default', learning_rate=0.001, batch_size=64, shuffle_batches=True, to_one_hot=False, n_classes=None, trainable_vars=None, restore=True, op_name=None, validation_monitors=None,) # TODO ??? # TODO accuracy
# 'adam' (Adaptive Moment Estimation)

# define the model
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='/storage/users/Muciaccia/tflearn_logs/', checkpoint_path=None, session=None)

# training with gradient descent
model.fit(dataset.features, dataset.classes, validation_set=0.5, show_metric=True, n_epoch=100, batch_size=128, shuffle=True, snapshot_epoch=True, snapshot_step=None, excl_trainops=None, validation_batch_size=None, run_id=None, callbacks=[])
# TODO nominare run_id con la data UTC attuale

exit()

############################

# TODO provare senza la media sulle features

import numpy

# TODO BUG. non c'è un modo per separare gli argomenti di default delle funzioni e renderli modulari. ad esempio: specificare il valore di 'questo' solo se si è messo 'quello=True'

categorical_predictions = model.predict(dataset.features)
# TODO inverse-one-hot
discrete_categorical_predictions = numpy.round(categorical_predictions)
flatten_discrete_predictions = numpy.argmax(discrete_categorical_predictions, axis=1)

# reverse the one-hot (categorical) encoding
flatten_true_classes = numpy.argmax(dataset.classes.values, axis=1)
# TODO sembra che numpy adesso supporti la computazione out-of-memory

correctness_of_predictions = numpy.equal(flatten_discrete_predictions, flatten_true_classes)

# TODO controllare ordine
a = flatten_discrete_predictions
b = flatten_true_classes
confusion_matrix = numpy.zeros([2,2]).astype(int)
# TODO BUG nell'assegnazione di numpy (stesso problema di Iuri
for i in range(len(a)):
    confusion_matrix[a[i],b[i]] += 1

# TODO ottenere questi dati da tflearn o da tensorboard per generalizzare la pipeline (in modo da poter poi metterci in mezzo il classificatore convolutivo)

# TODO dense_classifier.py
# TODO convolutional_classifier.py


#Huawei P8 lite smart

