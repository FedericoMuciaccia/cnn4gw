
# Copyright (C) 2016  Federico Muciaccia (federicomuciaccia@gmail.com)
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


#note
#The right tool for an image classification job is a convnet. Since we only have few examples, our number one concern should be overfitting. Overfitting happens when a model exposed to too few examples learns patterns that do not generalize to new data, i.e. when the model starts using irrelevant features for making predictions.
#Your main focus for fighting overfitting should be the entropic capacity of your model --how much information your model is allowed to store.
#There are different ways to modulate entropic capacity. The main one is the choice of the number of parameters in your model, i.e. the number of layers and the size of each layer. Another way is the use of weight regularization, such as L1 or L2 regularization, which consists in forcing model weights to taker smaller values.
#In our case we will use a very small convnet with few layers and few filters per layer, alongside data augmentation (TODO) and dropout. Dropout also helps reduce overfitting, by preventing a layer from seeing twice the exact same pattern, thus acting in a way analoguous to data augmentation (you could say that both dropout and data augmentation tend to disrupt random correlations occuring in your data).


import keras

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Activation, Dropout, Flatten, Dense, ZeroPadding2D

from keras.layers.normalization import BatchNormalization

#from keras.layers.advanced_activations import PReLU

import numpy
import pandas
from matplotlib import pyplot

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## fix random seed for reproducibility
#seed = 1
#numpy.random.seed(seed)

# TODO vedere se salvare i dati come matrici sparse per risparmiare spazio

# import the dataset
data = numpy.load('./../../data/clean_data.npy')

number_of_samples, image_width, image_height, channels = data['image'].shape
image_shape = image_width, image_height, channels # rows, columns, channels if keras dim_ordering='tf'

train_set = data[data['validation']==0]
test_set = data[data['validation']==1]
train_images = train_set['image']
train_classes = train_set['class']
validation_images = test_set['image']
validation_classes = test_set['class']

#images = numpy.load('./clean data/images (all).npy')
#classes = numpy.load('./clean data/classes (all).npy')

# TODO correggere a valori non asintotici per agevolare la sigmoide finale
#classes[classes == 1] = 0.95 # 1 - K.epsilon
#classes[classes == 0] = 0.05 # 0 + k.epsilon

# TODO farsi fare immagini/peakmap/spettrogrammi quadrati con lati con potenze di 2

## data shuffle
#images, classes = shuffle(images, classes)
## serve fare il random shuffle poiché il validation set viene sempre preso dai dati finali
#
## reshuffle one more time :)
#images, classes = shuffle(images, classes)
## TODO controllare che questo random refhuffle di scikit-learn sia effettivamente a maximum entropy
#
## data preparation
## 600 signal + 600 noise = 1200 images 98x82 pixel
## samples, rows, columns, channels
#number_of_samples, image_width, image_height = images.shape
#channels = 1 # black and white images
#images = images.reshape(number_of_samples, image_width, image_height, channels) # TODO procedura assurda
## TODO il reshape di numpy dovrebbe essere in-place (più sensato)
##image_shape = image_width, image_height, channels # rows, columns, channels if keras dim_ordering='tf'

# TODO mettere 1-epsilon e 0+epsilon per far convergere la funzione costo

## data preprocessing
## i dati sono già normalizzati tra 0 e 1
## in futuro normalizzare direttamente gli spettrogrammi
#
## split the dataset in train and validation
#validation_percentage = 1/2#/6
#train_images, validation_images, train_classes, validation_classes = train_test_split(images, classes, test_size=validation_percentage)
#
## save the datasets
#numpy.save('./clean data/train_images (all).npy', train_images)
#numpy.save('./clean data/train_classes (all).npy', train_classes)
#numpy.save('./clean data/validation_images (all).npy', validation_images)
#numpy.save('./clean data/validation_classes (all).npy', validation_classes)
## TODO save only the validation set

#def transform_classes(classes):
#	# change the encodind for the classes array
#	# from 0 = noise, 1 = signal
#	# to (1,0) = noise, (0,1) = signal
#	# than avoid asymptotic values
#	old_classes = classes.astype(numpy.bool)
#	lenght = len(classes)
#	epsilon = 1e-2 # 0.01
#	new_classes = numpy.zeros((lenght, 2), dtype=numpy.float32)
#	new_classes[old_classes] = 0+epsilon, 1-epsilon
#	new_classes[numpy.logical_not(old_classes)] = 1-epsilon, 0+epsilon
#	return new_classes

#train_classes = transform_classes(train_classes)
#validation_classes = transform_classes(validation_classes)

epochs = 50#100#200#100#50#25
# TODO limite massimo con la memoria e col numero dei CUDA cores? (K20: 5GB, 2496 cores)
# TODO 600 sembra essere attualmente il massimo numero che non dà problemi di memoria con la configurazione attuale
batch_size = 880#730#600#300#256#100
#kernel_size = (3, 3)
#concurrent_filters = 32

# model definition
#model = Sequential()
# convolutional layer 32x3x3: 32 concurrent filters 3x3
#model.add(Convolution2D(concurrent_filters, *kernel_size, input_shape=image_shape)) # TODO init, border, regularizer
# TODO why 32 filters? TODO fare rete che trova il valore ottimale
# TODO fare in modo che le dimensioni equivalenti dell'ultimo kernel siano tali da poter coprire tutta la durata temporale attesa del segnale transiente (1 day?)
#model.add(Activation('relu'))

#model.add(Convolution2D(concurrent_filters, *kernel_size, input_shape=image_shape))
#model.add(Activation('relu'))

# TODO conteggio dei parametri

# blocco elementare (ripetuto 3 volte):
# {
# convolution
# relu
# pool
# }

#model.add(Convolution2D(concurrent_filters, *kernel_size))
#model.add(Activation('relu')) # TODO fare imparare la pendenza della sigmoide o della relu

#model.add(Convolution2D(concurrent_filters, *kernel_size)) # TODO ci sono anche i parametri di bias. trascurarli?
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.5))

#model.add(Convolution2D(concurrent_filters, *kernel_size))
#model.add(Activation('relu'))

#model.add(Convolution2D(concurrent_filters, *kernel_size))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

# TODO mettere altri dropout aggressivi per evitare l'overfitting
# TODO provare a mettere un singolo blocco inception
# TODO visualizzare tutti i vari filtri ai vari livelli e verificare che siano smooth e senza noise

#########################

model = Sequential()

# TODO provare il blocco Inception elementare

# TODO studiare il BatchNormalization()
# va messo sempre tra il layer convolutivo e la funzione di attivazione
# media 0 e varianza 1. la normalizzazione va messa tra lo strato lineare e quello non lineare, per facilitare le cose alla funzione di attivazione
#model.add(BatchNormalization(axis=3)) # image channel axis
# TODO secondo me c'è un bug con la BatchNormalization con keras
#W tensorflow/core/framework/op_kernel.cc:975] Invalid argument: You must feed a value for placeholder tensor 'keras_learning_phase' with dtype bool [...]
# probabilmente viene escluso in fase di test, ma senza effettuare alcuna correzione, portando ad un serio allargamento dei pesi e dunque a risultati scarsissimi sul test set

#model.add(Input(shape=image_shape)) # TODO
model.add(ZeroPadding2D(input_shape=image_shape))
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# TODO è corretto mettere border_mode='same'?
# gli spazi ai bordi vengono riempiti con degli zeri
# TODO border_mode per il pooling?

model.add(ZeroPadding2D()) # TODO implementare le condizioni periodiche a contorno
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D())
model.add(Convolution2D(16, 3, 3))
#model.add(Convolution2D(16, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# TODO si possono legare i 32 filtri solo coi loro corrispettivi del livello precedente? probabilmente non ha senso perché inibisce la corretta ramificazione delle rappresentazioni gruppali
model.add(ZeroPadding2D())
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D())
model.add(Convolution2D(nb_row=3, nb_col=3, nb_filter=16)) # TODO così è backend-independent (stesso codice per tensorflow e theano)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

## rete puramente convolutiva fino alla fine: un output per ogni classe
#model.add(Convolution2D(nb_row=3, nb_col=2, nb_filter=2))
##model.add(BatchNormalization())
#model.add(Activation('sigmoid'))
#model.add(Flatten())
##model.add(Dense(2))
##model.add(Activation('sigmoid'))

#model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

# TODO sigmoid VS softmax
# la cosa migliore sarebbe avere alla fine due neuroni per le due categorie e poi un layer softmax per far uscire le probabilità di appartenere ad una data classe

# plot a model summary to check the number of parameters
model.summary()
#print('parametri: ', model.count_params())

# TODO senza la BatchNormalization sembrano esserci problemi a mettere diversi layer convolutivi in sequenza dentro lo stesso blocco

# TODO ampio kernel iniziale? (TODO visualizzarli)
# TODO numero crescente di filtri per poter combinare le rappresentazioni sempre più complesse
# TODO pochi filtri sul finale? (solo 2)

# TODO provare regolarizzatori L2 ed L1

# TODO fare una specie di reinforcement learning (?), dove gli errori recidivi vengono fatti rivedere spesso e vengono penalizzati di più. crea overfitting?

# TODO plottare distribuzione dei pesi e dei bias ai vari layer
# e metterli in un notebook con lo slide muovibile col cursore
# TODO rete neurale che decide cosa fare del training di un'altra rete neurale, guardando questi plot; che decida come propagare l'errore sulla base della forma della distribuzione della funzione costo

#########################

# TODO il dropout è una sorta di data-augmentation (aggiungendo o sottraendo rumore) fatta ad un qualunque layer?

# TODO se metto lo sparsity constrain risparmio molto spazio in memoria?

# TODO visualizzare i filtri creati dai vari strati convolutivi

# unit = conv, relu, conv, relu, pool

# the model so far outputs 3D feature maps (height, width, features)
# this converts our 3D feature maps to 1D feature vectors
#model.add(Flatten())

#model.add(Dropout(0.5))
# ri-aggiunto questo strato di dropout, perché dai plot sembra essercene bisogno
# TODO oppure ha più senso aggiungerlo prima dello stato precedente, quello subito dopo il flatten?

#model.add(Dense(32)) # TODO L1 vs L2 regularization?
#model.add(Activation('relu'))
##model.add(PReLU()) # learnable activation function # TODO self-regularization?

##model.add(Dense(16))
##model.add(Activation('relu'))

#model.add(Dense(1))
#model.add(Activation('sigmoid'))

# TODO forse mancano le normalizzazioni intermedie dei pesi

# duplico il modello per fare una k-fold cross-validation con k=2
# per usare tutto il dataset: una volta facendo il training su una metà ed il validation sull'altra metà e poi facendo il viceversa con un modello separato
# TODO


adam = keras.optimizers.Adam()

# model compiling
model.compile(loss='binary_crossentropy', # OR 'categorical_crossentropy' OR 'mean_squared_error'
	optimizer=adam, #'adam', # OR 'rmsprop' OR 'sdg'
	metrics=['accuracy'])# 'categorical_accuracy', 'precision', 'recall'

# per tutte le funzioni di costo disponibili viene fatta la media su tutto il campione e non semplicemente la somma

# save untrained model on disk
model.save('./models/untrained_model.h5')

# TODO per avere un grafico del costo più liscio serve un algoritmo di discesa sensibile alla derivata seconda?

# TODO è possibile diminuire ulteriormente il learning rate adattivo dell'algoritmo ADAM?
# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# lr: float >= 0. Learning rate.
# beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
# epsilon: float >= 0. Fuzz factor.

# TODO ROC Curves: Like precision and recall, accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of these values.

# TODO classification error = 1 - accuracy?

# TODO
# a loro serve l'efficienza
#precision: A measure of a classifiers exactness.
#recall: A measure of a classifiers completeness
#ROC Curves: Like precision and recall, accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of these values.

# TODO
# altra possibile metrica per le prestazioni:
# numero di esempi sbagliati: numpy.rint(len(validation_dataset) * (1 - accuracy))
# così si capisce fin dove ci si può spingere, dato che il minimo in scala logaritmica è 1

# TODO diverso misclassification cost per le differenti categorie? (esempio: salmoni vs sardine)

# TODO loss e accuracy per il train set dovrebbero essere gli ultimi valori per ogni epoca e non una media tra i valori ottenuti nei vari mini_batch, altrimenti è ovvio che i risultati del train saranno sempre irrimediabilmente più alti di quelli del test. MA l'ultimo valore del train set è quello col mini_batch più piccolo, quindi quello con più alte incertezza e variabilità statistiche. come si risolve questo problema? si calcola la dimensione del mini-batch e si mette il più piccolo all'inizio?

# TODO dato che ho 100 dati nel mini_batch, posso calcolare la varianza dell'accuracy e dell'errore (anche quelli sono indicatori importanti, penso) e plottarli come strisce azzurrine attorno ai valori medi?
# TODO la varianza della accuracy si può anche fare con 100 training random e facendo un istogramma 2D?

## evaluate the model
#scores = model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# TODO fargli fare la prediction sui dati dove c'è il segnale ma io non lo vedo

# model training and testing
history = model.fit(train_images, train_classes, 
	batch_size=batch_size, # number of samples per gradient update
	nb_epoch=epochs, # number of evaluation of the entire dataset
	verbose=True, 
	validation_data=(validation_images, validation_classes), # OR validation_split=validation_percentage
	shuffle=True) #  train data shuffled at each epoch. validation data never shuffled

# save history on disk
train_history = pandas.DataFrame(history.history)
train_history.to_csv('./media/training_history.csv', index=False)
# TODO vedere numpy structured array and numpy.savez and np.asarray(test_dictionary)

# TODO visualize the decision boundaries

# TODO fare la prova usando soltanto i dati recenti, dove non ci sono problemi di segnali nulli e/o doppi

# save trained model on disk
model.save('./models/trained_model.h5')
# this hdf5 file contains:
# - the architecture of the model
# - the weights of the model
# - the training configuration (loss, optimizer)
# - the state of the optimizer, allowing to later resume training

# TODO hack per non fargli fare l'errore alla fine
import gc
gc.collect()

# save weights on disk after training
#model.save_weights('cnn_weights.h5')
#model.load_weights('cnn_weights.h5')

# TODO
#from keras.applications.inception_v3 import InceptionV3
#model = InceptionV3(weights='imagenet', include_top=False)
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# TODO introdurre i checkpoint per poter spezzare il training in più parti

# TODO vedere che succede ad addestrare la rete solo coi nuovi dati, dove non ci sono errori di classificazione artificiali sul segnale
# TODO pulire a mano il vecchio dataset

# Note that the variance of the validation accuracy is fairly high, both because accuracy is a high-variance metric and because we only use 800 validation samples. A good validation strategy in such cases would be to do k-fold cross-validation, but this would require training k models for every evaluation round.

# How does one add :
# a) K-fold cross validation
# b) checkpoint

# Checkpointing is as follows

# filepath="weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True,mode='max')
# callbacks_list = [checkpoint]

# TODO Using the bottleneck features of a pre-trained network: 90% accuracy in a minute
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# data la struttura molto particolare di questi dati, usare una rete preaddestrata non serve a molto





