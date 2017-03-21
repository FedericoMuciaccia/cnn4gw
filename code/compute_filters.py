
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



# visualization of the filters via gradient ascent in input space

# TODO siamo sicuri che bisogna massimizzare sui layer convolutivi e non invece sulla successiva funzione di attivazione nonlineare?

import numpy

import keras
from keras import backend as K

#from scipy.misc import imsave

# build the VGG16 network with ImageNet weights
#model = vgg16.VGG16(weights='imagenet', include_top=False)
truncated_model = keras.models.load_model('./models/trained_model.h5')
# rimuovere i layer non convolutivi in fondo
# (in modo che l'input possa avere dimensione qualsiasi)
#truncated_model.pop()
#truncated_model.pop()
#truncated_model.pop()

# TODO in realtà qua non sembra esserci bisogno di usare un modello troncato

#truncated_model.save('./models/truncated_trained_model.h5')

#truncated_model.summary()

#first_layer = model.layers[-1]
# this is a placeholder tensor that will contain our generated images
#input_img = first_layer.input

# this is the placeholder for the input images
input_img = truncated_model.input

# get the symbolic outputs of each "key" layer (they have unique names).
layer_dict = dict([(layer.name, layer) for layer in truncated_model.layers if 'convolution2d' in layer.name])

# the name of the layer we want to visualize
#layer_name = 'convolution2d_5'
#filter_index = 0

# dimensions of the generated pictures for each filter.
img_width = 98#128
img_height = 82#128
# TODO renderlo generale (o cun un input qualsiasi alla rete oppure prendendo i parametri direttamente dall'input fisso della rete

#loss = K.mean(layer_output[:, filter_index, :, :])

# compute the gradient of the input picture wrt this loss
#grads = K.gradients(loss, input_img)[0]

#grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
def normalize(x):
    # normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

# do gradient ascent in the input space, with regard to our filter activation loss

# run gradient ascent for 20 steps
#for i in range(20):
#    loss_value, grads_value = iterate([input_img_data])
#    input_img_data += grads_value * step

# extract and display the generated input

# convert a tensor into a valid image
# TODO keras or scikit-image preprocessing?
def deprocess_image(x): # TODO BatchNormalization?
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = numpy.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = numpy.clip(x, 0, 255).astype('uint8') # TODO ?
    return x

#@numpy.vectorize # TODO non è una vera vettorializzazione (è solo un for interno)
def find_filters(layer_name):
	#kept_filters = []
	imgs = []
	losses = []
	# TODO rendere l'algoritmo vettoriale e parallelo
	layer = layer_dict[layer_name]
	layer_output = layer.output
	number_of_filters = layer.output_shape[-1]
	# we only scan through the first 16 filters or less,
	number_of_filters_to_display = numpy.min([number_of_filters, 16])
	for filter_index in numpy.arange(number_of_filters_to_display):
		# define a loss function that maximize the activation of a specific filter in the layer considered
		loss = K.mean(layer_output[:, :, :, filter_index]) # 	TODO ?
		# TODO errore quadratico medio?
		
		# compute the gradient of the input picture with respect to this loss
		grads = K.gradients(loss, input_img)[0] # TODO ?
	
		# normalize the gradient
		grads = normalize(grads) # TODO BatchNormalization?
		
		# this function returns the loss and grads given the input picture
		iterate = K.function([input_img], [loss, grads]) # TODO dev=finirla fuori dal ciclo for?
		
		# TODO non ha più senso usare gli algoritmi di ottimizzazione già esistenti ma invertendo il segno alla funzione costo?
		# step size for gradient ascent
		step = 1. # TODO come si stabilisce questo parametro?
	
		# start from a gray image with some random noise # TODO gaussian random?
		input_img_data = numpy.random.random((1, img_width, img_height, 1))#3)) # TODO renderlo generale, con anche il channel imposto dalla rete
		input_img_data = (input_img_data - 0.5) * 20 + 128 # TODO ?
		
		# run gradient ascent for 20 steps # TODO
		for i in numpy.arange(50):#20
			loss_value, grads_value = iterate([input_img_data])
			input_img_data += grads_value * step
		
		#print('loss value:', loss_value)
		if loss_value <= 0.: # TODO ?
			# some filters get stuck to 0, we can skip them
			print('filter %d skipped' % (filter_index + 1))
			#pass #break
		
		# decode the resulting input image
		if loss_value > 0: # TODO
			img = deprocess_image(input_img_data[0])
			#kept_filters.append((img, loss_value))
			#print(img)
			imgs.append(img)
			#print(loss)
			losses.append(loss_value)
			print('filter %d processed' % (filter_index + 1))
	imgs = numpy.array(imgs, dtype=numpy.float32) # TODO meglio uint?
	losses = numpy.array(losses, dtype=numpy.float32)
	
	# sort images by loss
	# to highlight the most relevant ones (?)
	# the filters that have the highest loss are assumed to be better-looking.
	#kept_filters.sort(key=lambda x: x[1], reverse=True)
	imgs = imgs[numpy.flipud(losses.argsort())]
	
	return imgs


#img = input_img_data[0]
#img = deprocess_image(img)
#imsave('%s_filter_%d.png' % (layer_name, filter_index), img)


# TODO parallelizzarlo
for layer_name in layer_dict.keys():
	print(layer_name)
	imgs = find_filters(layer_name)
	numpy.save('.images/filters/{name}.npy'.format(name=layer_name), imgs)
	#numpy.save('./losses.npy', losses)

# TODO riprovare con i filtri della rete VGG16

