
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




#https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
#https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
#https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist_v1.ipynb
#https://keras.io/layers/about-keras-layers/


#https://keras.io/getting-started/functional-api-guide/#the-concept-of-layer-node
#https://keras.io/getting-started/functional-api-guide/#the-concept-of-layer-node
#http://stackoverflow.com/questions/5470161/python-can-load-modules-from-remote-server
#https://www.python.org/dev/peps/pep-0302/
#https://www.reddit.com/r/learnpython/comments/32grtr/is_it_possible_to_use_remote_import_in_python/
#https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/
#http://stackoverflow.com/questions/39088489/tensorflow-periodic-padding

import numpy

import matplotlib
# backend per poter lavorare in remoto senza il server X
# PRIMA di importare pyplot
matplotlib.use('SVG')
from matplotlib import pyplot

import keras
from keras import backend as K

from visualize_filters import make_mosaic

numpy.set_printoptions(precision=5, suppress=True)

model = keras.models.load_model('./models/trained_model.h5')

# K.learning_phase() is a flag that indicates if the network is in training or predict phase. It allow layer (e.g. Dropout) to only be applied during training
inputs = [K.learning_phase()] + model.inputs

conv1 = model.layers[0]
conv2 = model.layers[3]
conv3 = model.layers[6]
conv4 = model.layers[9]
conv5 = model.layers[12]

# TODO si può fare con model.predict() fermandosi a un certo layer?
_convout1_f = K.function(inputs, [conv1.output])
def convout1_f(X):
    # The [0] is to disable the training phase flag
    return _convout1_f([0] + [X])

# utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pyplot.imshow"""
    if cmap is None:
        cmap = matplotlib.cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)#0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pyplot.colorbar(im, cax=cax)

#i = 4600
#
## Visualize the first layer of convolutions on an input image
#X = X_test[i:i+1]
#
#pyplot.figure()
#pyplot.title('input')
#nice_imshow(pyplot.gca(), numpy.squeeze(X), vmin=0, vmax=1, cmap=matplotlib.cm.binary)

def OLD_make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    #nimgs = imgs.shape[0]
    #imshape = imgs.shape[1:]
    nimgs, imrow, imcol = imgs.shape
    imshape = imrow, imcol
    #nimgs = imgs.shape[2]
    #imshape = imgs.shape[:2]
    
    
    mosaic = numpy.ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=numpy.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in numpy.arange(nimgs):
        row = int(numpy.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

#pyplot.imshow(make_mosaic(numpy.random.random((9, 10, 10)), 3, 3, border=1))

# Visualize weights
#W = model.layers[0].W.get_value(borrow=True)
#W = model.layers[0].W.read_value()
#W = model.layers[0].W.value()
W, b = model.layers[1].get_weights()
#W = numpy.squeeze(W) # leva la profondità di 1 per il bianco-nero
W = numpy.transpose(W, axes=(3,0,1,2))
#W = numpy.transpose(W, axes=(2,0,1)) # permutare gli indici per riportare il tutto alla notazione di tensorflow invece che a quella di Theano
# TODO segnalare bug sul github di keras
print("W shape : ", W.shape)

# TODO scipy.misc.imsave('kernel_visualization.svg', make_mosaic(W, 6, 6))

# TODO new make_mosaic()

W = numpy.squeeze(W) # leva la profondità di 1 per il bianco-nero # TODO

pyplot.figure(figsize=(10, 10))
pyplot.title('conv1 weights')
nice_imshow(pyplot.gca(), OLD_make_mosaic(W, 4, 4), vmin=-0.5, vmax=0.5, cmap=matplotlib.cm.bwr) # TODO settando il bianco (o il verde) sullo 0
pyplot.savefig('./images/kernel_visualization.svg')
pyplot.close()

pyplot.hist(W.flatten(), bins=50)
pyplot.title('conv1 weights distribution')
pyplot.savefig('./images/weights_distribution.svg')
pyplot.close()

pyplot.hist(b, bins=50)
pyplot.title('conv1 bias distribution')
pyplot.savefig('./images/bias_distribution.svg')
pyplot.close()

## Visualize convolution result (after activation) # TODO massimizzare l'output dopo la funzione di attivazione
#C1 = convout1_f(X)
#C1 = numpy.squeeze(C1)
#print("C1 shape : ", C1.shape)
#
#pyplot.figure(figsize=(15, 15))
#pyplot.suptitle('convout1')
#nice_imshow(pyplot.gca(), make_mosaic(C1, 6, 6), cmap=matplotlib.cm.binary)

# TODO visualizzare l'input che massimizza i filtri, per capire che tipo di forme vengono cercate

# TODO fare istogramma di W e B ad ogni layer per verificare se effettivamente ci sia del covariance shift


