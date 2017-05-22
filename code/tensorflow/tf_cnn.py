
import tensorflow as tf

import numpy
import tflearn


# import the dataset
# TODO check how to efficiently read data in pure tensorflow
# TODO usare un formato dati più standard e direttamente leggibile in tensorflow
data = numpy.load('../../../GENERATED_clean_data.npy')
# TODO don't load everything in memory with huge datasets
# TODO trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

# TODO vedere se su tensorflow si può evitare il canale del grigio per le immagini biancoonero
number_of_samples, image_width, image_height, channels = data['image'].shape
image_shape = image_width, image_height, channels # rows, columns, channels

train_set = data[data['validation']==0]
test_set = data[data['validation']==1]
train_images = train_set['image']
train_classes = train_set['class']
validation_images = test_set['image']
validation_classes = test_set['class']

# TODO slim.dataset slim.data_decoder

# free some memory space # TODO tutto incluso in funzione read_data()
del data

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

# TODO libro deep learning, rete tensorflow, strutture dati tensorflow sequenziali dinamiche, esempi infiniti, rete distribuita

# training parameters
# learning_rate = 0.001 # TODO is the initial value?
# TODO learning rate that exponentially decays over time (updated at every epoch)
#iterations = 100000 # TODO epochs
batch_size = 128 # TODO massimo 512 per lo stochastic gradient descent (small-batch regime)
# TODO capire perché con 128 converge molto molto più velocemente che con 512
#display_step = 10

number_of_epochs = 10#50

# classifier parameters
rows, columns, channels = image_shape
number_of_classes = 2



# TODO not memory efficient
train_x = tf.constant(train_images)
test_x = tf.constant(validation_images)
# one_hot endcoding:
# 1 -> [0,1]
# 0 -> [1,0]
train_y = tf.one_hot(train_classes, number_of_classes, dtype=tf.float32)
test_y = tf.one_hot(validation_classes, number_of_classes, dtype=tf.float32)

# categorical (one_hot encoding)
train_classes = tflearn.data_utils.to_categorical(train_classes, number_of_classes).astype(numpy.float32)
validation_classes = tflearn.data_utils.to_categorical(validation_classes, number_of_classes).astype(numpy.float32)



# graph input
images = tf.placeholder(dtype=tf.float32, shape=[None, rows, columns, channels]) # TODO provare senza canali per il solo bianconero
# reshape input image
true_classes = tf.placeholder(tf.float32, shape=[None, number_of_classes]) # labels
# TODO ???

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
    convolutional_layers = 5
    output_features = [8,8,16,16,32]
    kernel_sizes = [[3,3],[3,3],[3,3],[3,3],[3,3]]
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
                             padding='same') # or 'valid'
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
                                    strides=[2,2], # TODO un semplice blurrig (
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

# TODO using ReLU activation functions, it's good to have a slightly positive initial bias in the initialization, to avoid "dead neurons"
# TODO forse contrasta con l'effetto della batch_normalization
# Otherwise, if `normalizer_fn` is None and a `biases_initializer` is provided then a `biases` variable would be created and added the hidden units.
#def bias_variable(shape):
#    # TODO vedere se introdurre noise anche nel bias
#    initial_value = tf.constant(0.1, shape=shape)
#    return tf.Variable(initial_value)

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
average_categorical_cross_entropy = tf.reduce_mean(categorical_cross_entropy)
# loss = average_categorical_cross_entropy
optimizer = tf.train.AdamOptimizer() # TODO exponentially decaying learnig rate
train_step = optimizer.minimize(average_categorical_cross_entropy)
# default Adam parameters are ok (example: learning_rate)

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

# TODO plottare errore per i singoli dataset

#while True:
#    sess.run(my_train_op)

writer = tf.summary.FileWriter(logdir='../../../tf_logs/')
# asynchronously updates the file contents. this allows a training program to call methods to add data to the file directly from the training loop, without slowing down training
#writer.add_graph(sess.graph)
#writer.add_summary() # TODO
#writer.add_session_log()
#writer.add_event()

#with tf.device('/cpu:0'):
trainop = tflearn.TrainOp(loss=average_categorical_cross_entropy,
                          optimizer=optimizer,
                          metric=accuracy,
                          batch_size=batch_size)

trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=0)

trainer.fit(feed_dicts={images: train_images, true_classes: train_classes}, val_feed_dicts={images: validation_images, true_classes: validation_classes}, n_epoch=number_of_epochs)#, show_metric=True)
# se si chiama trainer.fit() successivamente, l'addestramento riprende da dove lo si era lasciato
# TODO implementare il curriculum learning o il decadimento esponenziale del learning_rate

# TODO model metrics and validation plots con tf.summary
# TODO training in puro tensorflow

# TODO vedere tensorboard

tf.summary.scalar('cross_entropy', average_categorical_cross_entropy)
tf.summary.scalar('accuracy', accuracy)
# tf.summary.image('input', x_image, 3)
# merged_summary = tf.summary.merge_all()

# tf.train.basic_train_loop(supervisor, train_step_fn)
# tf.train.limit_epochs
# tf.train.shuffle_batch
# tf.train.Supervisor
# A training helper that checkpoints models and computes summaries. The Supervisor is a small wrapper around a `Coordinator`, a `Saver`, and a `SessionManager` that takes care of common needs of TensorFlow training programs.

# Checkpoints are binary files in a proprietary format which map variable names to tensor values. # TODO ???

# with sess.as_default():

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


    
