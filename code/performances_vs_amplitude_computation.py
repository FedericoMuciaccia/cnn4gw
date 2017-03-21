
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

import sklearn.metrics

import matplotlib
matplotlib.use('SVG')
from matplotlib import pyplot


from read_data import read_it
# TODO duplicato
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


df = pandas.read_csv('./images/k_fold_dataframe.csv')

is_train = numpy.random.randint(2, size=len(df)).astype(bool)
df['is_train'] = is_train
#columns: amplitude, true_class, path, subset, is_train

# train set
train_dataframe = df[df['is_train'] == True]

# validation set
validation_dataframe = df[df['is_train'] == False]

train_images, train_classes = get_images_and_classes(train_dataframe)

validation_images, validation_classes = get_images_and_classes(validation_dataframe)

model = keras.models.load_model('./models/untrained_model.h5')

epochs = 50
batch_size = 730
history = model.fit(train_images, train_classes, 
		batch_size=batch_size,
		nb_epoch=epochs,
		verbose=True, 
		#validation_data=(validation_images, validation_classes), 
		shuffle=True)
#train_history = pandas.DataFrame(history.history)

model.save('./models/new_trained_model.h5')

# alleggerire un po' la memoria
del df
del train_dataframe
validation_dataframe.drop('path', axis=1, inplace=True)
validation_dataframe.drop('is_train', axis=1, inplace=True)
validation_dataframe.drop('subset', axis=1, inplace=True)

predicted_classes = model.predict(validation_images)
predicted_classes = predicted_classes.reshape(validation_dataframe['true_class'].values.shape)
rounded_predicted_classes = numpy.rint(predicted_classes).astype(int)

validation_dataframe['predicted_class'] = predicted_classes
validation_dataframe['rounded_predicted_class'] = rounded_predicted_classes

possible_amplitudes = numpy.array([0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005]) * 1e-22

grouped_dataframe = validation_dataframe.groupby('amplitude')

performances = []
for amplitude, group in grouped_dataframe:
	true_classes = group['true_class'].values
	rounded_predicted_classes = group['rounded_predicted_class'].values
	binary_confusion_matrix = sklearn.metrics.confusion_matrix(true_classes, rounded_predicted_classes, labels=None, sample_weight=None)
	[[p0t0,p1t0],[p0t1,p1t1]] = binary_confusion_matrix
	precision = (p1t1)/(p1t1 + p1t0) # purity
	recall = (p1t1)/(p1t1 + p0t1) # efficiency
	performances.append([amplitude, precision, recall])

performances = pandas.DataFrame(performances, columns=['amplitude','purity','efficiency'])

performances.to_csv('./images/performances_vs_amplitude.csv', index=False)


