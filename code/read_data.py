
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



import scipy.io
from matplotlib import pyplot
import pandas
import skimage.io
import numpy
import os

data_dir = "./data/raw data/simulated signal on gaussian white noise background/"
image_dir = './data img/skimage tif/'

# MATLAB 5.0 MAT-file
# (Matlab version 7.3 files are hdf5 datasets)
#@numpy.vectorize # TODO non funziona la vettorizzazione
def read_it(file_path):
	
	#print(file_path)
	a = scipy.io.loadmat(file_path)
	#b = a['peaksSS']
	b = a['SUB_peaks'] # seguo la nomenclatura di Pia
#	if 'SIG' in file_path:
#		whereis = a['WHEREIS'] # giorno di inizio e fine del segnale

	# print(len(b[0]), len(b[1])) # il numero di punti neri ovviamente varia
	# print(min(b[0]), max(b[0])) # i limiti in x coincidono
	# print(min(b[1]), max(b[1])) # i limiti in y variano con la finestra scelta
	
	#c = b[0,:] - b[0,1]
	#d = b[1,:]
	
	x = b[0,:] - b[0,1]
	y = b[1]
	#y = y[y>0]
	
	#z = b[2]
	#df = pandas.DataFrame({'time':x, 'frequency':y, 'significance':z})
	df = pandas.DataFrame({'time':x, 'frequency':y})
	df = df[df.frequency > 0]
	
	x = df.time.values
	y = df.frequency.values
	
	# normalized significance
	#df.significance = (df.significance - df.significance.min())/(df.significance.max() - df.significance.min())
	##gray_colors = df.significance.apply(str).values
	##gray_colors = df.significance.astype('float16').apply(str).values
	##pyplot.scatter(x,y, s=z)
	#z = df.significance.values
	#pyplot.scatter(x,y, c=z, cmap='gray_r', marker='.', linewidth=0)
	
	# TODO c'è probabilmente un bug nella canalizzazione, perché mi sono accorto che alcuni pixel bianchi (circa 1 su mille) hanno un conteggio di 2 invece che 1
	
	# istogramma supposto puramente binario
	histogram_results = numpy.histogram2d(x, y, bins=[98,82])
	image_matrix = histogram_results[0]
	# print('dimensione immagine:', image_matrix.shape)
	
	do_pyplot = False
	if do_pyplot:
		filename = os.path.split(filepath)[-1]
		image_dir = './data img/pyplot jpg/'
		pyplot.figure()
		pyplot.title(filename)
		pyplot.xlabel('Days from the beginning')
		pyplot.ylabel('Frequency [Hz]')
		#pyplot.plot(c, d, '.k')
		pyplot.scatter(x,y, c='black', marker='.')
		#pyplot.scatter(x,y, c=z, cmap='gray_r', marker='.', linewidth=0)
		pyplot.xlim(min(x), max(x))
		pyplot.ylim(min(y), max(y))
		#pyplot.show()
		pyplot.savefig(image_dir + filename + ".jpg")
		pyplot.close()

	# float32 per compatibilità con le GPU
	# e per futura codifica di tutto lo spettrogramma in scala di grigio
	return image_matrix.astype(numpy.float32)

def plot_it(filename):
	# immagine fatta di 0 e 1
	image_matrix = read_it(data_dir + filename)
	# immagine a 8 bit in bianco e nero (0 e 255)
	# uint: unsigned integer
	image_matrix = image_matrix.astype(numpy.uint8)*255
	# tif: formato immagine non compresso
	skimage.io.imsave(image_dir + filename + '.tif', image_matrix)
	# TODO interpolation='nearest' su pyplot.imshow per non avere i pixel slavati
	# TODO attenzione che l'immagine risulta ruotata per la convenzione su righe e colonne
	# TODO atterzione che skimage vuole le immagini in float comprese tra -1 e +1

#signal_images = []
#noise_images = []
#for filepath in os.listdir(data_dir):
#	# matlab_files.append(scipy.io.loadmat(data_dir+filename))
#	if 'SIG' in filepath:
#		signal_images.append(read_it(filepath))
#	if 'NOISE' in filepath:
#		noise_images.append(read_it(filepath))
#	#plot_it(filepath)
# print(len(signal_images), len(noise_images))
# create the two dataset files
#numpy.save('my_VIRGO_dataset_signal.npy', signal_images)
#numpy.save('my_VIRGO_dataset_noise.npy', noise_images)

import operator
def allFilesIn(topFolder):
	fileList = []
	for folder, subFolders, files in os.walk(topFolder):
		subFolders.sort(key=operator.methodcaller("lower"))
		files.sort(key=operator.methodcaller("lower"))
		for item in files:
			fileList.append(os.path.join(folder, item))
	return fileList


amplitudes = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005]
images = []
classes = []
amps = []	
for amplitude in amplitudes: # TODO parallelizzare
	folder = data_dir + str(amplitude) + 'e-22/'
#folder = data_dir
#if True:	
	#images = []
	#classes = []
	for filepath in allFilesIn(folder): #os.listdir(data_dir):
		images.append(read_it(filepath))
		amps.append(amplitude * 1e-22)
		if 'SIG' in filepath:
			classes.append(1) # 1: noise and signal
		if 'NOISE' in filepath:
			classes.append(0) # 0: noise only
		#filename = os.path.split(filepath)[-1]
		#plot_it(filename)
images = numpy.array(images)
images = images.astype(numpy.float32)
classes = numpy.array(classes)
classes = classes.astype(numpy.float32)
amps = numpy.array(amps)
amps = amps.astype(numpy.float32)
	#images = numpy.array(images)
	#images = images.astype(numpy.float32)
	#classes = numpy.array(classes)
	#classes = classes.astype(numpy.float32)
	# TODO salvare gli array come 'sparse', dato che ci sono un sacco di zeri
	# TODO oppure salvare i dati in binario, per risparmiare spazio, e poi fare un preprocessing delle immagini prima di fare i calcoli
	#print(images.shape, classes.shape)
#	numpy.save('./clean data/images (all).npy', images)
#	numpy.save('./clean data/classes (all).npy', classes)
#	numpy.save('./data/images ({amp}e-22).npy'.format(amp=amplitude), images)
#	numpy.save('./data/classes ({amp}e-22).npy'.format(amp=amplitude), classes)

####################################

# data preparation and preprocessing

from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split

# data shuffle
images, classes, amps = shuffle(images, classes, amps)
# serve fare il random shuffle poiché il validation set viene sempre preso dai dati finali

# reshuffle one more time :)
images, classes, amps = shuffle(images, classes, amps)
# TODO controllare che questo random refhuffle di scikit-learn sia effettivamente a maximum entropy

# data preparation
# 600 signal + 600 noise = 1200 images 98x82 pixel
# samples, rows, columns, channels
number_of_samples, image_width, image_height = images.shape
channels = 1 # black and white images
images = images.reshape(number_of_samples, image_width, image_height, channels) # TODO procedura assurda
# TODO il reshape di numpy dovrebbe essere in-place (più sensato)
image_shape = image_width, image_height, channels # rows, columns, channels if keras dim_ordering='tf'

# TODO mettere 1-epsilon e 0+epsilon per far convergere la funzione costo

# data preprocessing
# i dati sono già normalizzati tra 0 e 1
# in futuro normalizzare direttamente gli spettrogrammi

# split the dataset in train and validation
validation = numpy.random.random_integers(low=0, high=1+1, size=len(images)) # piccolo hack per avere l'intervallo chiuso a destra
#validation_percentage = 1/2#/6 # TODO generalizzare
#train_images, validation_images, train_classes, validation_classes = #train_test_split(images, classes, test_size=validation_percentage)

###############################

# create a numpy structured array
clean_data = numpy.array(list(zip(images,classes,amps,validation)), dtype='(98,82,1)float32, float32, float32, int8')
clean_data.dtype.names = ('image','class','amplitude','validation')
# save the data on disk
numpy.save('./data/clean_data.npy', clean_data)
# TODO usare un formato file più universalemte riconosciuto e leggibile di npy (csv? hdf5? h5?)

exit()

#time = np.linspace(0,1.25,50)
#omega = np.linspace(0,2*np.pi,50)
#times, omegas = np.meshgrid(time,omega)

#dt = 0.047407

#pyplot.hist2d(x, y, bins=[98,41], cmap='gray_r')
#pyplot.colorbar()
#numpy.histogram2d()

#for matlab_file in matlab_files:
#	read_and_plot(matlab_file)


# con 98 canali in tempo (x) non si hanno doppi conteggi in 2D
# con 41 canali in frequenza (y) non si hanno doppi conteggi in 2D
# ---------------
# con 98 canali si ha canalizzazione 1D continua nel tempo (x)
# con 82 canali si ha canalizzazione 1D continua in frequenza (y)
# ---------------
# dunque immagine 98*82 pixel?
# cosa rappresenta 8110? valori troncati di poco

	
# intervallo di frequenze: da 0 a 0.01 Hz
# intervallo di tempi: 4.5 days
# campionamento?
# noise molto più fitto che nei file col segnale !!!
# alcuni file di segnale non lo hanno visibile !!! (errore di Pia)
# facendo io l'istogramma 2D lei mi può dare direttamente gli array per lo scatterplot come prima
# se lei mi dà le misure esatte posso fare anche un semplice reshape in forma rettangolare
# levare dalla cartella quella ventina di file che non c'entrano nulla
# non ci sono file a intensità 1/3
# non riesco a usare nè theano nè tensorflow nè il multicore e sono attualmente obbligato a far girare tutto sulla mia CPU, con tempi molto dilatati
# ci sono alcuni dati sbagliati ma la macchina mi dice accuracy = 1 (magari non sono finiti nel validation set?)
# magari per ora concentrarsi sul ridurre l'intensità del segnale, invece che ingrandire la finestra. vedere fino a che rapporto segnale/rumore ci si riesce a spingere
# dopo che si arriva alla loro sensibilità, applicare la rete direttamente allo spettrogramma e non alla peakmap
# ancora non è una rete profonda e riesco a farla girare sul mio processore (circa 4 minuti, con questo dataset molto piccolo)
# dopo aver ridotto i segnali sul white noise, provare coi segnali sul rumore vero

# i file ad ampiezza 1/10 non presentano doppi e tripli segnali. questo può creare problemi di riconoscimento (adesso invece tutti i files hanno un segnale solo)
# la finestra in frequenza non è duplicata
# i files sono ancora 8110 e non nel formato di prima
# i dati a grandi intensità presentano ancora il problema dei segnali mancanti. li ho ripuliti a mano. quelli nuovi sembrano a posto
# nei prossimi giorni terrò questo dataset ma cambierò l'architettura, rendendola un po' più ricca
# aggiungere solo i files a intensità 1/100

# i file così sono semplici e perfetti per cominciare
# a naso si può cominciare anche con segnali meno intensi
# TODO conteggio dei punti per la RAM
# TODO poi in futuro provare direttamente col segnale a monte dello spettrogramma (per fargli imparare una DFT nonlineare ad adattiva sui dati)


