
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

import keras

import os

# TODO duplicato
import operator
def allFilesIn(topFolder):
	fileList = []
	for folder, subFolders, files in os.walk(topFolder):
		subFolders.sort(key=operator.methodcaller("lower"))
		files.sort(key=operator.methodcaller("lower"))
		for item in files:
			fileList.append(os.path.join(folder, item))
	return fileList

data_dir = "./raw data/simulated signal on gaussian white noise background/"

k_fold_number = 10
paths = []
true_classes = []
amplitudes = []
k_fold_subset = []
possible_amplitudes = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005]
for amplitude in possible_amplitudes:
	folder = data_dir + str(amplitude) + 'e-22/'
	#images = []
	

	#amplitude = amplitude * 1e-22
	for filepath in allFilesIn(folder): #os.listdir(data_dir):
		paths.append(filepath)
		#images.append(read_it(filepath))
		if 'SIG' in filepath:
			true_classes.append(1) # 1: noise and signal
		if 'NOISE' in filepath:
			true_classes.append(0) # 0: noise only
		#filename = os.path.split(filepath)[-1]
		#plot_it(filename)
		amplitudes.append(amplitude*1e-22)
		k_fold_subset.append(numpy.random.randint(k_fold_number))
	#images = numpy.array(images)
	#images = images.astype(numpy.float32)
	#classes = numpy.array(classes)
	#classes = classes.astype(numpy.float32)
	# TODO salvare gli array come 'sparse', dato che ci sono un sacco di zeri
	# TODO oppure salvare i dati in binario, per risparmiare spazio, e poi fare un preprocessing delle immagini prima di fare i calcoli
	#print(images.shape, classes.shape)
	#numpy.save('./clean data/images (all).npy', images)
	#numpy.save('./clean data/classes (all).npy', classes)
df = pandas.DataFrame({'path':paths, 'true_class':true_classes, 'amplitude':amplitudes, 'subset':k_fold_subset})
possible_amplitudes = numpy.array(possible_amplitudes) * 1e-22

# 10-fold
# dato che tutti i vari sottodataset sono divisibili per 10

# data shuffle
# two times, in two different ways :)
df = df.sample(frac=1)
df = df.reindex(numpy.random.permutation(df.index))

df.to_csv('./media/k_fold_dataframe.csv', index=False)

from read_data import read_it

def get_images_and_classes(dataframe):
	images = []
	for image_path in dataframe.path.values:
		image = read_it(image_path)
		images.append(image)
	classes = dataframe['true_class'].values
	images = numpy.array(images)
	classes = numpy.array(classes)
	images = images.astype(numpy.float32)
	classes = classes.astype(numpy.float32)

	# TODO duplicato
	number_of_samples, image_width, image_height = images.shape
	channels = 1
	images = images.reshape(number_of_samples, image_width, image_height, channels)

	return images, classes


epochs = 50
batch_size = 600#100
validation_percentage = 1/2
total_history = numpy.zeros((k_fold_number, epochs, 4)) # 4 poiché nella history ci sono accuracy e loss sia per il train che per il test
for i in range(k_fold_number):
	print("running fold", i+1, "/", k_fold_number)
	validation_dataframe = df[df.subset == i]
	validation_images, validation_classes = get_images_and_classes(validation_dataframe)
	train_dataframe = df[df.subset != i]
	train_images, train_classes = get_images_and_classes(train_dataframe)
	model = None
	model = keras.models.load_model('./models/untrained_model.h5')
	history = model.fit(train_images, train_classes, 
		batch_size=batch_size,
		nb_epoch=epochs,
		verbose=True, 
		validation_data=(validation_images, validation_classes), 
		#validation_split=validation_percentage, 
		shuffle=True)
	train_history = pandas.DataFrame(history.history)
	# be sure that the columns are in alphabetical order
	train_history = train_history[sorted(train_history.columns)]
	total_history[i] = train_history.values

# NOTA: questo calcolo prende circa 50 minuti su GPU e 4.5 GB di memoria grafica

numpy.save('./media/k_fold history.npy', total_history)



