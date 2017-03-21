
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



import numpy

import scipy
import scipy.misc

#import keras

#filters = pandas.DataFrame(kept_filters, columns=['image', 'loss'])
# TODO pandas è lentissimo se gli si mette dentro matrici (tipo immagini): sostituirlo con un database (esempio: coppia immagine-costo)
#filters.imgs

#@numpy.vectorize
#def zero_pad(img, margin=margin):
#	return numpy.pad(img, pad_width=margin, mode='constant', constant_values=0)
# TODO pandas.apply(zero_pad())
#imgs = zero_pad(imgs)

# NOTA ho testato l'algoritmo con un mosaico di foto vere e funziona perfettamente: risultano correttamente orientate dritte e sono perfettamente ordinate in maniera crescente da sinistra verso destra e dall'alto verso il basso, esattamente come per la scrittura su un foglio di carta

def make_mosaic(imgs, rows, cols, margin=1):
	# all 2D images must have the same shape
	
	# TODO fare casi separati per immagini bianconero, bianconero con convenzione tensorflow, rgb ed rgb-alpha
	
	samples = len(imgs)
	if samples <= rows * cols:
		# prepare extra empty space to fill the grid
		empty_images = rows*cols - samples
	if samples > rows * cols:
		empty_images = 0
		# only keep the top filters
		# discard the other (less relevant) images
		imgs = imgs[0: rows * cols]
	
	# set padding sizes (before and after) for every dimension of the tensor
	number_padding = (0,empty_images)
	height_padding = (margin, margin)
	width_padding = (margin, margin)
	channel_padding = (0,0)
	padding_settings = (number_padding, height_padding, width_padding, channel_padding)
	
	# zero padding of the images
	imgs = numpy.pad(imgs, padding_settings, mode='constant', constant_values=0)
	
	samples, height, width, channels = imgs.shape
	# samples is now equal to rows * cols
	# let's rearrange things to create a grid of images
	mosaic = imgs.reshape(rows, cols, height, width, channels)
	# permutation of some indices
	mosaic = numpy.transpose(mosaic, axes=(0,2,1,3,4))
	mosaic = mosaic.reshape(rows*height, cols*width, channels)

	# add a zero padding as external border
	mosaic = numpy.pad(mosaic, (height_padding, width_padding, channel_padding), mode='constant', constant_values=0)
	return mosaic

if __name__ == "__main__":
	#truncated_model = keras.models.load_model('truncated_trained_model.h5')
	# TODO generalizzare automaticamente a tutti i layer convolutivi presenti
	#layer_names = [layer.name for layer in truncated_model.layers if 'convolution2d' in layer.name]
	layer_names = ['convolution2d_{number}'.format(number=n) for n in numpy.arange(1,5+1)]
	for layer_name in layer_names: # TODO parallelizzare
		imgs = numpy.load('./images/filters/{name}.npy'.format(name=layer_name))
		#losses = numpy.load('./losses.npy')
		
		rows, cols = 4, 4 #5, 6
		margin = 2
		
		mosaic = make_mosaic(imgs, rows, cols, margin)
		
		# if the image is black and white
		if mosaic.shape[-1] == 1:
			mosaic = numpy.squeeze(mosaic) # leva la profondità di 1 per il bianco-nero perché così vuole scipy.misc.imsave # TODO
	
		# TODO vedere scikit-image
		scipy.misc.imsave('./images/filters/{name}.tif'.format(name=layer_name), mosaic) # TODO png OR svg?
		# TODO rigirare le immagini di 90 gradi, in modo da fare un mosaico con la stessa orienzazione delle vere immagini di input
	
	# TODO riprovare con i filtri della rete VGG16



