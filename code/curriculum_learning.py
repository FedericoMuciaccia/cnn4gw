
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
import pandas
import sklearn
import keras

# TODO farlo coi checkpoints/callbacks

amplitudes = [0.3, 0.1, 0.05, 0.01, 0.008, 0.005]
amplitudes = pandas.DataFrame({'amplitude':amplitudes})
factors = [1, 0.1, 0.1, 0.01, 0.01, 0.01]
#epochs = [30,30, 50]

# TODO chiedere a Pia altri dati con queste ampiezze
# 0.09 0.08 0.07 0.06 0.04 0.03 0.02
# 0.009 0.007 0.006 0.004 0.002 0.001

model = keras.models.load_model('./models/untrained_model.h5')

# TODO duplicato (tranne la fine)
def preprocess(images):
	number_of_samples, image_width, image_height = images.shape
	channels = 1
	images = images.reshape(number_of_samples, image_width, image_height, channels)
	return images

validation_percentage = 1/2

epochs = 25
batch_size = 300
# TODO poi nelle ultime fasi, quando il training è più difficile, aumentarlo a 600

#images = []
#classes = []
train_history = pandas.DataFrame()
#for amplitude in amplitudes:
for index, amplitude in amplitudes.itertuples(): # non parallelizzabile: apprendimento sequanziale
	images = numpy.load('./data/images ({amp}e-22).npy'.format(amp=amplitude))
	classes = numpy.load('./data/classes ({amp}e-22).npy'.format(amp=amplitude))
	
	images = preprocess(images)
	
	if index == amplitudes.index.max():
		break
	next_amplitude = amplitudes.amplitude[index + 1]
	
	new_images = numpy.load('./data/images ({amp}e-22).npy'.format(amp=next_amplitude))
	new_classes = numpy.load('./data/classes ({amp}e-22).npy'.format(amp=next_amplitude))
	
	new_images = preprocess(new_images)
	
	
	images = numpy.concatenate([images, new_images])
	classes = numpy.concatenate([classes, new_classes])
	
	images, classes = sklearn.utils.shuffle(images, classes)
	
	# TODO il fine tuning va fatto con SGD (tunando learning rate, momentum e decay?)
	#sgd = keras.optimizers.SGD(lr=0.1, momentum=0.5)
	#model.compile(loss='binary_crossentropy',
	#optimizer=sgd,
	#metrics=['accuracy'])
	
	default_adam_learning_rate = 0.001
	#factor = 1/2**index
	factor = factors[index]
	learning_rate = factor * default_adam_learning_rate
	#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	
	model.optimizer.lr.assign(learning_rate)
	#model.optimizer.decay.assign(0.5)
	
	history = model.fit(images, classes, 
		batch_size=batch_size,
		nb_epoch=epochs,
		verbose=True, 
		validation_split=validation_percentage, # OR validation_data=(validation_images, validation_classes)
		shuffle=True)
	
	new_train_history = pandas.DataFrame(history.history)
	train_history = pandas.concat([train_history, new_train_history], ignore_index=True)
	
	
train_history.to_csv('./images/training_history.csv', index=False)

