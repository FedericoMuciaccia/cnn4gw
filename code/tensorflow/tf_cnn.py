
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


import tensorflow as tf

import numpy
#import tflearn # TODO BUG? confligge con netCDF4

import time

import xarray

# TODO aumentare numero immagini (out-of-memory)
# TODO mettere batch_normalization
# TODO provare con la mediana invece che la media
# TODO calcolare a mano valore iniziale (in base e si ha: log(2) = 0.69314718055994529)
# TODO inizializzare bias e pesi leggermente in positivo
# TODO servirebbero almeno 4800 immagini
# TODO nei vecchi dati forse c'è un problema di learning rate (molti rimbalzi alti)
# TODO provare keras coi dati vecchi e coi dati nuovi in singolo canale

# TODO BUG in python: 7j == 0 + 7*1j (j non è definita, che sarebbe la cosa più logica)

# import the dataset
# TODO check how to efficiently read data in pure tensorflow
# TODO usare un formato dati più standard e direttamente leggibile in tensorflow
#data = numpy.load('../../../GENERATED_clean_data.npy')
# TODO don't load everything in memory with huge datasets
# TODO trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

dataset = xarray.open_dataset('/storage/users/Muciaccia/images.netCDF4')#, chunks={'sample_index': 128}) # TODO è meglio lasciare che il chunck_size venga gestito internamente in maniera ottimale (il training risulta parecchio più veloce)

# dataset.images[0:128].values.nbytes # slice(128)
# ogni chunk da 128 immagini è grosso circa 100MB
# (il chunksize raccomandato in Dask è tra 10 ee 100 MB)
# dunque è anche un buon valore da usare come batch_size

# TODO BUG di xarray (aggiunge channel anche a is_noise_only)
#dataset = dataset.where(dataset.channel == 'red')

# TODO BUG di xarray: su jupyter-console quando si preme Tab per vedere l'autocompletamento dei metodi di un dataset, questo viene inutilmente caricato tutto in memoria, anche se ha un chunksize ridotto (e poi molta memoria viene subito dopo liberata, ma il fatto di essere stata temporaneamente massicciamente occupata alle volte questo crea problemi a jupyter-console fino addirittura ad interrompergli il kernel di esecuzione).

#dataset.images.where(dataset.is_noise_only & dataset.is_for_validation, drop=True)
train_images = dataset.images.where(numpy.logical_not(dataset.is_for_validation), drop=True) #.values
validation_images = dataset.images.where(dataset.is_for_validation, drop=True) #.values
# TODO chunks di 128 o 64 immagini


#dataset.sel(sets = 'validation', drop=True)

# TODO vedere se la nuova versione di numpy supporta nativamente i Dask array

# dataset.images
# x = dataset.images[300:500].values
# del x
# Dataset.filter_by_attrs(**kwargs)
# train_images.chunk(chunks=[batch_size, rows, columns, channels]).chunks

# dict(my_key='my_value')
# >>> {'my_key': 'my_value'}
# dict(key_1='value_1', key_2='value_2')
# >>> {'key_1': 'value_1', 'key_2': 'value_2'}

# TODO mettere nomi migliori
# gli array delle classi sono già in categorical encoding (one-hot encoding) in modo da essere già in formato Dask per la computazione out-of-memory (che altrimenti non si potrebbe fare usando tflearn.data_utils.to_categorical) (anche se in effetti lo spazio occupato in memoria è così poco che ce ne si poteva fregare)
train_classes = dataset.classes.where(numpy.logical_not(dataset.is_for_validation), drop=True) # .values
validation_classes = dataset.classes.where(dataset.is_for_validation, drop=True) # .values
# TODO data generator (out-of-memory)?

# TODO senza usare dataset.load() si occupano massimo 3GB di RAM invece che più di 8GB (quasi 10, considerando la anche la swap), MA l'utilizzo della RAM è estremamente oscillante (segno che c'è molta interazione col disco di memoria) ed inoltre il tempo di training risulta doppio (proprio forse a causa del collo di bottiglia del disco di memorizzazione). se invece NON si specifica a mano un chunk_size, l'utilizzo di RAM è di leggermente superiore al caso di prima (ora massimo 5GB) ma molto più regolare e meno oscillante. il training inoltre, paradossalmente ed inspiegabilmente, è anche leggermente più veloce di quando si carica tutto in RAM (circa il 10% in meno).
# TODO vedere se le stesse differenze reggono utilizzando la GPU
#train_images.load()
#train_classes.load()
#validation_images.load()
#validation_classes.load()
# TODO ha senso usare una rappresentazione compressa dei dati mediante la rete Inception già addestrata?

# TODO plottare degli esempi di imagini dalle due classi, in modo da capire se sono posizionate correttamente

# TODO rendere ['noise', 'noise+signal'] una dimensione
# TODO rendere ['train', 'validation'] una dimensione

# classifier parameters
sample_number, rows, columns, channels = dataset.images.shape
number_of_classes = 2

# training parameters
# learning_rate = 0.001 # TODO is the initial value?
# TODO learning rate that exponentially decays over time (updated at every epoch)
#number_of_iterations = 500#100000 # TODO epochs
batch_size = 128#64 # TODO massimo 512 per lo stochastic gradient descent (small-batch regime) # TODO provare anche 64
# TODO capire perché con 128 converge molto molto più velocemente che con 512 (forse gli outlier disturbano la media? mettere la mediana?)
#display_step = 10

number_of_epochs = 100#50#10#50


#random_order = np.arange(len(X))
#np.random.shuffle(random_order)
#X, y = X[random_order], y[random_order]

# # use the old dataset
# data = numpy.load('/storage/users/Muciaccia/OLD_clean_data.npy')
# 
# sample_number, rows, columns, channels = data['image'].shape
# 
# train_set = data[data['validation']==0]
# test_set = data[data['validation']==1]
# train_images = train_set['image']
# train_classes = train_set['class']
# validation_images = test_set['image']
# validation_classes = test_set['class']

# TODO vedere se su tensorflow si può evitare il canale del grigio per le immagini biancoonero
#number_of_samples, image_width, image_height, channels = data['image'].shape
#image_shape = image_width, image_height, channels # rows, columns, channels

#train_set = data[data['validation']==0]
#test_set = data[data['validation']==1]
#train_images = train_set['image']
#train_classes = train_set['class']
#validation_images = test_set['image']
#validation_classes = test_set['class']

#N = 1000
#image_shape = [1024, 128, 3]

# TODO use tflearn dask out-of-memory dataset

#train_images = numpy.single(numpy.random.rand(N, *image_shape))
#train_classes = numpy.single(numpy.round(numpy.random.rand(N)))
#validation_images = numpy.single(numpy.random.rand(int(N/100), *image_shape))
#validation_classes = numpy.single(numpy.round(numpy.random.rand(int(N/100))))

# TODO slim.dataset slim.data_decoder

# free some memory space # TODO tutto incluso in funzione read_data()
#del data

# TODO data augmentation?

# TODO ZCA whitening and PCA whitening
# https://en.wikipedia.org/wiki/Whitening_transformation
# add_zca_whitening (pc=None)

# TODO decidere Sì/No e poi, se Sì, evidenziare un quadratino attorno al segnale

# NOTE per Ricci:
# - facendo girare su CPU non si hanno vincoli di RAM
# - è possibile usare immagini anche molto grandi anche con GPU
# - aversial networks per simulare e caratterizzare il rumore (segnale = non vero rumore?)
# - 

# TODO libro deep learning,strutture dati tensorflow sequenziali dinamiche, esempi infiniti, rete distribuita

# TODO far vedere le immagini sale e pepe del white noise e le immagini complete RGB per dare un'idea di quanto il compito sia più difficile

# TODO not memory efficient
#train_x = tf.constant(train_images)
#test_x = tf.constant(validation_images)

# one_hot endcoding:
# 1 -> [0,1]
# 0 -> [1,0]
#train_classes = tf.one_hot(train_classes, number_of_classes, dtype=tf.float32)
#validation_classes = tf.one_hot(validation_classes, number_of_classes, dtype=tf.float32)
# TODO capire perché in tflearn non si possono passare a trainer.fit(...) dei tensori di TensorFlow ma si è obbligati ad usare array di numpy, che occupano molta memoria

# categorical (one_hot encoding)
#train_classes = tflearn.data_utils.to_categorical(train_classes, number_of_classes).astype(numpy.float32)
#validation_classes = tflearn.data_utils.to_categorical(validation_classes, number_of_classes).astype(numpy.float32)
# TODO BUG su numpy non c'è il to_categorical, come invece c'è in Matlab

#df = train_classes.to_dataframe(name='classe') 
#cdf = df.classe.astype('category') 
#cdf.cat.categories = ['noise', 'noise+signal']

# graph input
images = tf.placeholder(dtype=tf.float32, shape=[None, rows, columns, channels])
true_classes = tf.placeholder(dtype=tf.float32, shape=[None, number_of_classes]) # labels
# a placeholder exists solely to serve as the target of feeds. it is not initialized and contains no data

#dropout_probability = 0.75 # probability that a neuron's output is kept during the learning phase
#keep_probability = tf.placeholder(tf.float32, shape=[]) # TODO tf.constant

## create some wrappers for simplicity
#def convolutional_block(x, W, b):
#    """usage: x = convolutional_block(x, weights['conv{}'.format(layer+1)], biases['conv{}'.format(layer+1)])"""
#    # TODO tf.contrib.layers.conv2d or tf.slim.conv2d
#    # x is an image
#    strides = 1
#    # TODO zero_padding
#    # convolve the image with the weights tensor
#    # TODO (N kernels, with N the number of output features in the layer)
#    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
#    # add the bias
#    x = tf.nn.bias_add(x, b) # TODO tf.add
#    # TODO batch_normalization
#    x = tf.nn.relu(x)
#    # TODO dropout
#    return x

# create model
# TODO with tf.Graph().as_default():
def neural_network(images): #, weights, biases):
    # rename the input
    x = images
    # TODO tf.input
    
    # convolutional part
    # TODO tf.slim.repeat
    convolutional_layers = 4
    #output_features = [8,8,16,16,32]
    output_features = [9,9,9,9]
    kernel_sizes = [[3,3],[3,3],[3,3],[3,3]]
    # TODO tf.contrib.slim.repeat
    # TODO vedere cos'è che fa parire il loss da livelli pazzeschi (tipo 10000)
    for layer in range(convolutional_layers): # TODO vedere se il for di python rallenta tutto
        # cross-correlation
        # 2D_convolution + bias
        # la convoluzione è in reltà qui una cross-correlazione
        # TODO vedere qual'è il significato con Kadhanoff blocking e renormalization group
        x = tf.layers.conv2d(x,
                             filters=output_features[layer],
                             kernel_size=kernel_sizes[layer],
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                             # TODO valori di varianza maggiori (come ad esempio il default stddev=1.0) portano il valore iniziale del loss a livelli astronomici (esempio: 10000) invece di circa 0.69, che penso debba essere il valore naturale
                             # TODO kernel_initializer = tf.contrib.layers.variance_scaling_initializer(...)
                             # the truncated_normal distribution stops at 2 sigmas
                             use_bias=True, # TODO use a batch_normalization function instead of `biases`
                             bias_initializer = tf.zeros_initializer(), # if None, no bias will be applied
                             # TODO mettere piccoli valori positivi (randomici?) per evitare i 'neuroni morti' con la successiva ReLU
                             #kernel_regularizer=None,
                             #bias_regularizer=None, 
                             #activity_regularizer=None, # TODO regularizer function for the output
                             #dilation_rate = [1,1] # dilation rate to use for a'trous convolution (dilated convolution). TODO Kadhanoff
                             # specifying any `dilation_rate` value != 1 is incompatible with specifying any stride value != 1
                             #activation=None, # linear activation
                             strides=[1,1],
                             padding='valid') # or 'same'
                             # or tf.nn.conv2d
                             # or tf.contrib.layers.convolution2d
        
#        # batch normalization to reduce internal covariate shift
#        # TODO vedere articolo. training mode (statistics of the current batch) or in inference mode (moving statistics)
#        # reference: http://arxiv.org/abs/1502.03167
#        x = tf.layers.batch_normalization(x,
#                                          #axis=-1,
#                                          #momentum=?,
#                                          center=False, # if True, add offset of `beta` to normalized tensor # TODO add a bias
#                                          scale=False, # if True, multiply by `gamma`
#                                          #beta_initializer=tf.zeros_initializer(),
#                                          #gamma_initializer=tf.ones_initializer(),
#                                          #moving_mean_initializer=tf.zeros_initializer(),
#                                          #moving_variance_initializer=tf.ones_initializer(),
#                                          #beta_regularizer=None,
#                                          #gamma_regularizer=None, 
#                                          trainable=False) # TODO default: trainable=True
#                                          # TODO learnable: different variance normalizations in the Kadhanof blocking?
#                                          # or tf.contrib.layers.batch_norm(...)
#                                          # or tf.nn.batch_normalization(...)
#                                          # or tf.contrib.slim.batch_norm(...)
        
        # activation function (ReLU)
        x = tf.nn.relu(x) # phi(x) = max(x,0)
        
        # down-sampling (max pooling)
        x = tf.layers.max_pooling2d(x,
                                    pool_size=[2,2],
                                    strides=[2,2], # TODO un semplice blurrig
                                    padding='same') # or 'valid'
                                    # or tf.nn.max_pool(...)
                                    # or tf.contrib.layers.max_pool2d(...)
        
#        # try to prevent overfitting (dropout)
#        dropout_probability = 0.5 # TODO tuning e definizione fuori
#        x = tf.layers.dropout(x, rate=dropout_probability) # rate=0.1 would drop out 10% of input units
#        # or tf.nn.dropout(keep_probability)
    
    # flatten the output of the convolutional part
    x = tf.contrib.layers.flatten(x) # the input is a tensor of size [batch_size, ...]
    # or tf.reshape(x, [-1, weights['dense1'].get_shape().as_list()[0]]
    # TODO pass '[-1]' to flatten 't': reshape(t, [-1])
    # TODO cercare ove possibile di usare solo tensorflow puro ed eliminare tutti i pazzi derivanti da tf.contrib
    
    # fully-connected part
    # readout layer
    # Wx+b with no activation function
    logit_output = tf.layers.dense(x, # x is a flat array of neurons
                        units=number_of_classes,
                        activation=None, # no softmax activation here: read below to know why
                        use_bias=True,
                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), # TODO valutare valore ottimale varianza
                        bias_initializer=tf.zeros_initializer(), # TODO errato?
                        kernel_regularizer=None, # TODO
                        bias_regularizer=None,
                        activity_regularizer=None) # TODO ??
                        # or tf.contrib.layers.fully_connected
    # the last activation function (softmax) is here omitted because it's computed later together with the cross entropy, in a more numerically-stable way
    # so the outputs don't represent class probabilities (they can be negatives, from -inf to +inf, and don't sum up to 1). they are called "logits"
                        
#    W, b = weights['dense1'], biases['dense1']
#    logit_output = tf.add(tf.matmul(x, W), b) # Wx+b
    
    return logit_output

## initialize weights with a small amount of noise (for symmetry breaking and to prevent null gradients)
#def weight_variable(shape):
#    initial_value = tf.truncated_normal(shape, stddev=0.1)
#    # TODO truncated_normal vs random_normal con bassa varianza
#    return tf.Variable(initial_value)

# TODO capire come si fa esattamente la backpropagation con le reti convolutive

# TODO using ReLU activation functions, it's good to have a slightly positive initial bias in the initialization, to avoid "dead neurons"
# TODO forse contrasta con l'effetto della batch_normalization
# Otherwise, if `normalizer_fn` is None and a `biases_initializer` is provided then a `biases` variable would be created and added the hidden units.
#def bias_variable(shape):
#    # TODO vedere se introdurre noise anche nel bias
#    initial_value = tf.constant(0.1, shape=shape)
#    return tf.Variable(initial_value)

# convolutional_kernel_shape = [kernel_size, kernel_size, input_features, number_of_filters]

# store weights ad biases
#weights = {
#    'conv1': weight_variable([3,3,1,8]), # 3x3 kernel, 1 input, 8 output
#    'conv2': weight_variable([3,3,8,8]), # TODO ???
#    'conv3': weight_variable([3,3,8,16]), # TODO non sembra esserci alcun vantaggio di memoria nello stringere l'immagine con i pool
#    'conv4': weight_variable([3,3,16,16]),
#    'conv5': weight_variable([3,3,16,32]),
#    'dense1': weight_variable([4*3*32,number_of_classes]), # TODO generalizzare TODO ??? ([5*5*32, 10]),
#}

#biases = {
#    'conv1': bias_variable([8]),
#    'conv2': bias_variable([8]),
#    'conv3': bias_variable([16]),
#    'conv4': bias_variable([16]),
#    'conv5': bias_variable([32]),
#    'dense1': bias_variable([number_of_classes])
#}

# construct model
logit_predictions = neural_network(images) #, weights, biases)

# define the cost function and the optimizer
# TODO binary_cross_entropy
# TODO ???
# Computes softmax cross entropy between logits and labels.
# Measures the probability error in discrete classification tasks in which the classes are mutually exclusive (each entry is in exactly one class).
# compute softmax and cross entropy together, to take care of numerical stability
# logits values are not probabilities
# the real loss/cost function is the categorical cross entropy
# The raw formulation of cross-entropy,
# tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
#                               reduction_indices=[1]))
# can be numerically unstable
categorical_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_predictions, labels=true_classes) # TODO capire logits, inversa sbagliata in sigmoide, softmax coi negativi
# average across the batch
average_categorical_cross_entropy = tf.reduce_mean(categorical_cross_entropy) # TODO median VS mean
# loss = average_categorical_cross_entropy
# learning_rate
optimizer = tf.train.AdamOptimizer() # TODO exponentially decaying learnig rate

# creates a variable to hold the global_step
global_step = tf.Variable(0, trainable=False, name='global_step')

train_step = optimizer.minimize(average_categorical_cross_entropy, global_step=global_step) # train_op
# default Adam parameters are ok (example: learning_rate)



# TODO importalo dopo: workaround per BUG su netCDF4 che non comprendo
import tflearn # TODO capire perché l'importazione è così irragionevolmente lenta

#with tf.device('/cpu:0'):
trainop = tflearn.TrainOp(loss=average_categorical_cross_entropy,
                          optimizer=optimizer,
                          #metric=accuracy,
                          batch_size=batch_size)

trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=0)

trainer.fit(feed_dicts={images: train_images, true_classes: train_classes}, val_feed_dicts={images: validation_images, true_classes: validation_classes}, n_epoch=number_of_epochs)#, show_metric=True)
# TODO assicurarsi che ci sia lo shuffle ad ogni epoca
# TODO snapshot_step=None, snapshot_epoch=True, shuffle_all=None, dprep_dict=None, daug_dict=None, excl_trainops=None, run_id=None, callbacks=[]
# TODO Log directory: /tmp/tflearn_logs/
# se si chiama trainer.fit() successivamente, l'addestramento riprende da dove lo si era lasciato
# TODO implementare il curriculum learning o il decadimento esponenziale del learning_rate

# TODO model metrics and validation plots con tf.summary
# TODO training in puro tensorflow

# TODO vedere tensorboard


exit()




#class AttrDict(dict):
#    def __init__(self, *args, **kwargs):
#        super(AttrDict, self).__init__(*args, **kwargs)
#        self.__dict__ = self
#
#FLAGS = AttrDict({'data_directory': '/storage/users/Muciaccia/'})

# TODO TensorFlow input pipelines with very large datasets (out-of-memory computation)
with tf.Graph().as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #images_batch, labels_batch = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
        images_batch, labels_batch = tf.train.batch([train_images,train_classes], batch_size, enqueue_many=True, allow_smaller_final_batch=True) # l'argomento va comunque dentro una lista, anche se è un singolo grosso tensore # TODO mettere shuffle
        # TODO  The value of a feed cannot be a tf.Tensor object. Acceptable feed values include Python scalars, strings, lists, numpy ndarrays, or TensorHandles.
        
        for step in range(number_of_iterations):
            begin_time = time.time()
            sess.run(train_step, feed_dict={images: images_batch, true_classes: labels_batch})
            end_time = time.time()
            duration = end_time - begin_time
            loss_value = sess.run(loss, feed_dict={images: images_batch, true_classes: labels_batch})
            
            if step % 10 == 0:
                # print an oveerview on screen
                print('iteration:', step)
                print('loss:', loss_value)
                print('global_step:', tf.train.global_step(sess, global_step_tensor))
                print('duration:', duration)
                
                # write the summary




# compute class prediction with softmax
# TODO duplicato
class_probabilities = tf.nn.softmax(logit_predictions)
# P_0 e P_1 sommano correttamente a 1
predicted_classes = tf.cast(tf.argmax(tf.round(class_probabilities)), tf.float32) # TODO
# rounding with threshold at 0.5
# 0.49 -> 0.0
# 0.50 -> 0.0
# 0.51 -> 1.0

# i neuroni finali devono essere 2 (uno per ogni classe, invece che un singolo neurone con sigmoide) perché così la rete può essere facilmente ampliata in futuro

# evaluate the model
# TODO rounded_predictions or discrete_predictions or class_predictions
# TODO predicted_classes and true_classes
correctness_of_predictions = tf.cast(tf.equal(predicted_classes, true_classes), tf.float32)
# TODO tf.argmax(Y, 1)
# TODO accuracy = tf.metrics.accuracy(true_classes, predicted_classes)
accuracy = tf.reduce_mean(correctness_of_predictions)
# accuracy: how often prediction matches true_class in the whole dataset
# mean(float(predicted_classes == true_classes))
# TODO tf.confusion_matrix, precision and recall

classification_error = 1 - accuracy # TODO check

#classification_error.eval(feed_dict={image:validation_images})

# TODO plottare errore per i singoli dataset

#while True:
#    sess.run(my_train_op)

writer = tf.summary.FileWriter(logdir='/storage/users/Muciaccia/tf_logs/')
# asynchronously updates the file contents. this allows a training program to call methods to add data to the file directly from the training loop, without slowing down training
#writer.add_graph(sess.graph)
#writer.add_summary() # TODO
#writer.add_session_log()
#writer.add_event()

######################################


def save_tfrecords_file(file_path, dataset):
    images = dataset.images
    classes = dataset.classes
    number_of_samples, rows, columns, channels = images.shape
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(number_of_samples):
        label = int(classes[index])
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            #'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[rows])),
            #'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[columns])),
            #'channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
            'class': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=dataset.images.shape)), 
            'image_flatten_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}))
        writer.write(example.SerializeToString())
    writer.close()
# TODO la dimensione è assolutamente simile a quella usata da numpy.save

# TODO provare a fare il feeding dei batch ditettamente da xarray

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())


tflearn.datasets
ds = tensorflow.contrib.learn.datasets.base.Dataset(data=x, target=y)
ds.data
ds.target

ds = tf.contrib.learn.datasets.base.Datasets(train=A, validation=B, test=C)

# TFRecords

images
classes

# TODO legare la lettura del chunks del netCDF4 al feeding del batch size

name = 'train' # 'validation'

import os

filename = os.path.join(FLAGS.data_directory, name + '.tfrecords')


# tf.train.Feature
# tf.train.Features
# tf.train.FeatureList
# tf.train.FeatureLists

# BytesList, FloatList, Int64List




# Avi code:


# PAGAMENTO LAUREA ENTRO 60 GIORNI DALL'APPELLO
# VERBALIZZARE TIROCINIO E CORSO_MONOGRAFICO


# list of filenames
files = tf.train.match_filenames_once('/storage/users/Muciaccia/netCDF4/O2/C01/128Hz/*.netCDF4')
# filename queue
filename_queue = tf.train.string_input_producer(files, num_epochs=None, shuffle=True, seed=None, shared_name=None, name=None) # FIFOQueue
# a reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

for i in range(1000):
  batch = ds.train.next_batch(128)

  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={images: batch[0], true_classes: batch[1]})
  
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



######################################

tf.summary.scalar('cross_entropy', average_categorical_cross_entropy) # loss.op.name
tf.summary.scalar('accuracy', accuracy)
# tf.summary.image('input', x_image, 3)
# merged_summary = tf.summary.merge_all()

# tf.train.basic_train_loop(supervisor, train_step_fn)
# tf.train.limit_epochs
# tf.train.shuffle_batch
# tf.train.Supervisor
# A training helper that checkpoints models and computes summaries. The Supervisor is a small wrapper around a `Coordinator`, a `Saver`, and a `SessionManager` that takes care of common needs of TensorFlow training programs.
# To train with replicas you deploy the same program in a `Cluster`.

# Checkpoints are binary files in a proprietary format which map variable names to tensor values. # TODO ???

# with sess.as_default():

exit()

# TODO forse con la nuova versione ci si può fermare già qui

# launch the graph
with tf.Session() as sess: # TODO with
    # initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    step = 1 # first iteration
    # Keep training until reach max iterations
    # TODO tf.while_loop

    writer.add_graph(sess.graph)
    exit()
    
    # iterations = number of batches used in the training
    # epochs = number of times that the whole dataset is used during the training
    while step * batch_size < iterations: # number of training examples shown to the neural network
        # the steps argument sets the number of mini-batches to train on
        # TODO vs 50 epochs with shuffled data
        #train_dataset = nmist.train
        #batch_images, batch_classes = train_dataset.next_batch(batch_size)
        batch_images, batch_classes = tf.train.batch([train_images,train_classes], batch_size, enqueue_many=True, allow_smaller_final_batch=True) # l'argomento va comunque dentro una lista, anche se è un singolo grosso tensore
        print(1)
        batch_xs, batch_ys = sess.run([batch_images, batch_classes])
        # TODO imageBatch, labelBatch = tf.train.shuffle_batch([image, label], batch_size=100, capacity=2000, min_after_dequeue=1000)
        # TODO Operations vs Tensors. Session.run() Tensor.eval()
        # backpropagation (optimization, gradient descent)
        print(2)
        train_step.run(feed_dict={images: batch_xs,
                                        true_classes: batch_ys#,
                                        #keep_probability: 0.75
                                        })
        print(3)
        if step % display_step == 0:
            # calculate loss and accuracy in the minibatch # TODO history
            minibatch_loss, minibatch_accuracy = sess.run([cost, accuracy], feed_dict={x: batch_images,
                                                              y: batch_classes,
                                                              keep_probabiity: 1.0}) # to make dropout layers transparent during performance evaluation
            print("Iter " + str(step*batch_size))
            print("minibatch loss:", minibatch_loss) # TODO "{:.6f}".format(minibatch_loss)
            print("training accuracy:", minibatch_accuracy) # TODO "{:.5f}".format(minibatch_accuracy)
        step += 1
    print("optimization finished!")

    # calculate validation accuracy
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images, # TODO accuracy: tensorflow variable vs ordinary function
                                                   y: validation_classes#,
                                                   #keep_probability: 1.0
                                                   }) # to make dropout layers transparent during performance evaluation # TODO learning_phase training_phase learnable flags
    print("validation accuracy:", validation_accuracy)



# shuffle array at unison
x, y = tflearn.data_utils.shuffle(x, y)

# tflearn.data_utils.load_csv (filepath, target_column=-1, columns_to_ignore=None, has_header=True, categorical_labels=False, n_classes=None)
# tflearn.data_utils.image_preloader (target_path, image_shape, mode='file', normalize=True, grayscale=False, categorical_labels=True, files_extension=None, filter_channel=False)
# tflearn.data_utils.build_hdf5_image_dataset (target_path, image_shape, output_path='dataset.h5', mode='file', categorical_labels=True, normalize=True, grayscale=False, files_extension=None, chunks=False)
# tflearn.data_utils.to_categorical (y, nb_classes)
# # binary vectors (to be used with 'categorical_crossentropy')

# # Build neural network and train
# network = ...
# model = DNN(network, ...)
# model.fit(X, Y)

# TODO valutare se usare direttamente una libreria di alto livello come tflearn o tensorflow puro

for epoch in range(training_epochs):
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))


    
