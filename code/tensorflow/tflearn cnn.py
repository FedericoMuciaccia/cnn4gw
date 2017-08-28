
import numpy
import xarray

# data loading and preprocessing

dataset = xarray.open_dataset('/storage/users/Muciaccia/images.netCDF4')

train_images = dataset.images.where(numpy.logical_not(dataset.is_for_validation), drop=True)
validation_images = dataset.images.where(dataset.is_for_validation, drop=True)

train_classes = dataset.classes.where(numpy.logical_not(dataset.is_for_validation), drop=True)
validation_classes = dataset.classes.where(dataset.is_for_validation, drop=True)

sample_number, rows, columns, channels = dataset.images.shape
sample_number, number_of_classes = dataset.classes.shape



import tflearn

# build the convolutional network
network = tflearn.layers.core.input_data(shape=[None, rows, columns, channels], name='input')
for i in range(5):
    network = tflearn.layers.conv.conv_2d(network, nb_filter=9, filter_size=3, strides=1, padding='same', activation='relu', bias=True, weights_init='uniform_scaling', bias_init='zeros', regularizer='L2', weight_decay=0.001) # activation='linear', regularizer=None, scope=None
    network = tflearn.layers.conv.max_pool_2d(network, kernel_size=2) # strides=None, padding='same'
    #network = tflearn.layers.normalization.local_response_normalization(network)
network = tflearn.layers.core.flatten(network)
#network = tflearn.layers.core.dropout(network, 0.8)
network = tflearn.layers.core.fully_connected(network, n_units=number_of_classes, bias=True, weights_init='truncated_normal', bias_init='zeros', activation='softmax') # regularizer=None, weight_decay=0.001, scope=None
network = tflearn.layers.estimator.regression(network, optimizer='adam', learning_rate=0.01, batch_size=128, loss='categorical_crossentropy', name='target') # metric='default', learning_rate=0.001, shuffle_batches=True, to_one_hot=False, n_classes=None, validation_monitors=None

# TODO mettere learning_rate, levare regolarizzatori, controllare input, mettere solo un neurone finale, mettere flatten, mettere batch_size, controllare feed_dict nell'altra rete, separare script per il preprocessing dell'input, aumentare profondità fino alla fine della memoria, provare ad aumentare il segnale (mettendolo a 1), controllare summary della rete, mettere relu, sistemare weight_decay, mettere normalizzazione al posto corretto

# training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input':train_images}, {'target':train_classes}, n_epoch=50, validation_set=({'input':validation_images}, {'target':validation_classes}), snapshot_step=100, show_metric=True, run_id='tflearn_conv_net_trial')

