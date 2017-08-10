
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

# TODO BUG: dovrebbero esserci automaticamente delle variabili di sistema come 'copyright' e 'license' che si riempono da sole quando specifico la licenza che desidero per il mio programma. ad esempio si rienpe da solo il campo `mioprogramma --license`. oppure proprio specificare la licenza tramite queste variabili. questo software potrebbe controllare in automatico le licenze delle librerie che uso come dipendenze ed avvisarmi automaticamente di potensiali conflitti tra le licenze (ad esempio se sto usando una libreria la cui licenza non è compatibile con la licenza GPL usata dal mio programma). serve definire uno standar. esempio: 
# copyright.year = 2017
# copyright.owner = 'Federico Muciaccia'
# copyright.email = 'federicomuciaccia@gmail.com'
# copyright.address = None
# copyright.license = 'GPLv3' # oppure 'all-rights-reserved' o 'public-domain' o 'C0'

import numpy
import xarray
import matplotlib
matplotlib.use('SVG') # per poter girare lo script pure in remoto sul server, dove non c'è il server X
from matplotlib import pyplot

RGB_images = numpy.load('/storage/users/Muciaccia/background_RGB_images.npy') # TODO file di 4 GB
# TODO mettere tutto su xarray perché serve un database completo con tutti i label/classi e gli attributi

number_of_samples, rows, columns, channels = RGB_images.shape

# esempio di immagine col solo background
# lo salvo adesso per usarlo dopo, per plottarlo a seguito della log-normalizzazione
RGB_noise_example = RGB_images[0].copy()

# TODO mettere artificialmente dei buchi nei dati di Virgo, giusto per far vedere tutte le possibili combinazioni di colore
#RGB_noise_example[:,slice(25,45),2] = 0
#RGB_noise_example[:,slice(70,95),2] = 0

def log_normalize(image):
    # log-normalization of the images
    # final RGB pixels must be a float from 0 to 1
#    image[image < 0] = 0 # per evitare problemi col gaussian white noise
    # TODO controllare i livelli di minimo e massimo del rumore
    log_minimum = 20 # TODO col segno meno (x_min = 1e-20)
    log_maximum = 10 # TODO col segno meno (x_max = 1e-10)
    log_image = numpy.log(image) + log_minimum
    log_image = log_image/log_maximum
    # TODO BUG in python/numpy non c'è un modo elegante per rappresentare le parentesi nelle espressioni matematiche senza che vengano create inutili tuple
    # TODO gestire gli infiniti senza rovinare il white noise
    #log_image[numpy.isinf(log_image)] = 0
    # mette a 0 i valori residuali minori di 0 e mette a 1 quelli residuali maggiori di 1
    log_image = numpy.clip(a=log_image, a_min=0, a_max=1)
#    log_image[log_image < 0] = 0 # -inf < 0 = True
#    log_image[log_image > 1] = 1 # TODO dopo il whitening non ci dovrebbero più essere problemi di dati fuori scala
    #pyplot.hist(log_image.flatten(), bins=100)
    #pyplot.show()
    return log_image
#log_normalize = numpy.frompyfunc(log_normalize, 1,1) # TODO hack per vettorializzare la funzione # TODO perché il decoratore @numpy.vectorize non funziona bene

def plot_RGB_image(RGB_image, file_path=None, kwargs={}):
    log_image = log_normalize(RGB_image)
    # TODO se già si è log-normalizzato l'intero dataset, si può usare direttamente train_images[0,:,:,0].plot();pyplot.show()
    pyplot.figure(figsize=[10,10*256/148]) # TODO 256x128
    fig = pyplot.imshow(log_image, origin="lower", interpolation="none", **kwargs)
    extent = fig.get_window_extent().transformed(pyplot.gcf().dpi_scale_trans.inverted()) # TODO attenzione che il formato dell'immagine non sembra quello corretto SENZA imporre prima il figsize. possibile errore di fondo?
    if file_path is not None:
        pyplot.savefig(file_path, dpi=300, bbox_inches=extent)
        pyplot.close()
    else:
        pyplot.show()

# funzione usata solo per plottare l'istogramma dei valori non riscalati
def safe_logarithm(image):
    log_image = numpy.log(image)
    # avoid log(0)=-inf
    log_image[numpy.isneginf(log_image)] = 0
    return log_image

# log_normalize(RGB_noise_example[:,:,0].flatten())
# take the first 100 background images to make an istogram of their values with enough statistic
R = safe_logarithm(RGB_images[0:100,:,:,0].flatten()) # 0.6 +- 0.1
G = safe_logarithm(RGB_images[0:100,:,:,1].flatten())
B = safe_logarithm(RGB_images[0:100,:,:,2].flatten())

pyplot.figure(figsize=[15,10])
pyplot.title('background pixel values distribution')
#pyplot.xlim([0,1])
pyplot.xlim([-20,-10])
pyplot.xlabel('logarithm of not-zero background pixel values')
pyplot.ylabel('count')
pyplot.hist([R,G,B], 
            bins=250,
            #range=[0,1],
            range=[-20,-10],
            label=['H O2 C01',
                   'L O2 C01',
                   'V VSR4 (shifted)'], # TODO imporre ordine RGB nella legenda (ordine di plot)
            color=['red','green','blue'],
            histtype='step')
            #linewidth=2,
            #fill=True,
            #alpha=0.1)
pyplot.vlines(x=numpy.log(1e-6), ymin=0, ymax=40000, color='black', label='level of the injected signal (1e-6)')
pyplot.legend(loc='upper right', frameon=False)
pyplot.savefig('/storage/users/Muciaccia/media/background_histograms.svg')
#pyplot.show()
pyplot.close()
# TODO farlo in scala semilog, che si capisce meglio ed è un formato molto più comune # TODO vedere se le due forme funzionali sono esattamente le stesse con lin_dati+log_scala VS log_dati+lin_scala
# TODO vedere se si può un pochino aumentare il contrasto dell'immagine centrando meglio gli estremi di minimo e massimo
# TODO valutare se normalizzare l'istogramma


# TODO rifare l'istogramma in scala non riscalata, in modo da capire veramente a che livello sono i dati e di che numeri si sta parlando

# TODO controllare dappertutto i logaritmi in base e o in base 10

# TODO libro neuroscienze: Kandel

# inietto direttamente un pattern nel dominio delle frequenze
# una sinusoide troncata con un certo spindown appare nello spettrogramma semplicemente come un segmento inclinato verso il basso

def add_signal(image, signal_intensity = 1e-5): # un segnale di 1e-5 si vede benissimo ad occhio nudo, con un errore sostanzialmente nullo. 1e-6 si vede ancora ma non benissimo. 0.7e-6 è l'ultimo valore per cui si riesce a vedere (veramente a stento) ad occhio nudo, poiché si trova sul picco delle gaussiane dell'istogramma dei pixel. sotto il picco non si riesce ad andare. il livello di 1e-6 mi sembra comunque inferiore alla soglia di 2.5 sigma che si mette per le peakmap, quindi comunque questo classificatore arriva sotto il loro limite di detectability. il classificatore denso NON riesce a riconosce il livello di 1e-6
    rows, columns, channels = image.shape
    
    max_spindown = 16
    frequency_border = 2*max_spindown
    time_border = 16
    
    def half(value):
        return int(value/2)
    
    # creo lo spazio dei parametri
    # TODO per il momento tutto misurato in unità di pixels
    # TODO in seguito fare tutto coi veri valori dei tempi e delle frequenze e non i valori degli indici/pixel
    signal_starting_frequency = numpy.random.randint(low=0+frequency_border, high=rows-frequency_border)
    signal_starting_time = numpy.random.randint(low=0+time_border, high=half(columns)-time_border)
    signal_ending_time = numpy.random.randint(low=half(columns)+time_border, high=columns-time_border)
    signal_spindown = numpy.random.randint(low=-max_spindown, high=0)
    # the minus sign is because we want signals with decreasing frequency (spindown and not spinup)
    # spindown = frequency_difference = f_end - f_start
    # example: f_end = 3, f_start = 5, spindown = -2
    signal_ending_frequency = signal_starting_frequency + signal_spindown
    
    # y = m*x + b
    frequency_difference = signal_spindown
    time_difference = signal_ending_time - signal_starting_time
    m = frequency_difference / time_difference
    b = - m*signal_starting_time + signal_starting_frequency
    
    t = numpy.arange(signal_starting_time, signal_ending_time)
    f = numpy.round(m*t+b).astype(int)
        
    flagged_values = numpy.equal(image, 0)
    image[f, t] += signal_intensity
    
    # le modifiche si ripercuotono direttamente sul tensore RGB_images perché queste sono views e non copie
    # TODO per poi fare signal-to-noise ratio
    # TODO in futuro l'ampiezza del segnale potrà anche decadere nel tempo
    # TODO inoltre questo segnale appare ugualmente intenso in tutti i detector, che invece hanno sensibilità diverse
    image[flagged_values] = 0 # fa in modo che il segnale non ci sia dove/quando il relativo detector risulta spento
    # TODO inserire normalizzazione

# TODO creare un generatore che crea direttamente un'immagine col solo segnale (funzione random_signal che restituisce un'immagine, magari salvata come matrice sparsa per risparmiare spazio)
# TODO BUG: su numpy manca una buona implementazione delle matrici sparse
# TODO BUG: vedere sintassi in numpy.arange?

# TODO salvarne metà dati di puro noise
# half the dataset will be noise only
is_noise_only = numpy.round(numpy.random.rand(number_of_samples)).astype(bool)
has_signal = numpy.logical_not(is_noise_only)
# half the dataset will be for validation
is_for_validation = numpy.round(numpy.random.rand(number_of_samples)).astype(bool)
# TODO BUG non c'è un modo più immediato per generare una sequenza di booleani casuale

for i in range(number_of_samples):
    if has_signal[i] == True:
        add_signal(RGB_images[i])

## TODO parallelizzare/vettorializzare questo ciclo for
#def inject_signal(images):
#    for i in range(len(images)):
#        add_signal(images[i])
## TODO BUG: NON FUNZIONA (forse perché le funzioni di python generano copie e non views)
#inject_signal(RGB_images[has_signal])

#i = 0
#for image in RGB_images[0:10]:
#    plot_RGB_image(image, '/storage/users/Muciaccia/media/prova/{}.jpg'.format(i))
#    i += 1

# TODO BUG: per un linguaggio moderno dovrebbe essere facile parallelizzare e vettorializzare anche se lunghezze e tipi di dati su cui operare sono disomogenei. esempio: square([scalare, vettore, matrice]) dovrebbe funzionare tranquillamente senza problemi

#    if i < 100:
#        plot_RGB_image(image, '/storage/users/Muciaccia/media/example_images/{}.jpg'.format(i)) 


# TODO capire perché in tutte le imagini spunta una linea orizzontale magenta a circa due terzi

# valori in grigio medio (usando una distribuzione log-normale)
#post = numpy.random.normal(loc=0.5, scale=0.1, size=1000)
#pyplot.hist(post, bins=100)
#pyplot.show()
#pre = numpy.exp(10 * post - 20)
#pyplot.hist(log_normalize(pre), bins=100)
#pyplot.show()
#pyplot.hist(pre, bins=100)
#pyplot.show()

# plot an example of noise-only image
plot_RGB_image(RGB_noise_example, '/storage/users/Muciaccia/media/RGB_noise_example.jpg')

# plot an example of simulated signal
i = 0
empty_image = numpy.zeros_like(RGB_images[i])
#empty_image[y[i], x[i]] = signal_intensity
empty_image += 1
add_signal(empty_image)
empty_image -= 1
signal_example = empty_image
plot_RGB_image(signal_example, '/storage/users/Muciaccia/media/simulated_signal_example.jpg')

# plot an example of background+signal image
flagged_values = RGB_noise_example == 0
final_image_example = RGB_noise_example + signal_example
final_image_example[flagged_values] = 0
plot_RGB_image(final_image_example, '/storage/users/Muciaccia/media/simulated_signal_on_RGB_noise_example.jpg')

# plot the single color components
plot_RGB_image(final_image_example[:,:,0], '/storage/users/Muciaccia/media/red_only.jpg', kwargs={'cmap':'Reds_r'})
plot_RGB_image(final_image_example[:,:,1], '/storage/users/Muciaccia/media/green_only.jpg', kwargs={'cmap':'Greens_r'})
plot_RGB_image(final_image_example[:,:,2], '/storage/users/Muciaccia/media/blue_only.jpg', kwargs={'cmap':'Blues_r'})
# TODO vedere se si riesce a mettere il nero come colore basale per lo zero

# TODO ereditare tutto dai precedenti dataset
coordinate_names = ['sample_index','rows','columns','channels']
coordinate_values = {'channels':['red','green','blue']}

images = xarray.DataArray(data=log_normalize(RGB_images).astype(numpy.float32), # TODO attenzione a non fare il logaritmo due volte, se per caso viene già fatto prima nel ciclo for dove si inietta il segnale # TODO vedere perché esce in float64
                          dims=coordinate_names, 
                          coords=coordinate_values)

#not_noise_only = xarray.DataArray(data=not_noise_only,
#                                  dims=['sample_index'])

def one_hot_encoding(array, number_of_classes):
    category_index = array.astype(int)
    categorical = numpy.eye(number_of_classes, dtype=numpy.float32)[category_index]
    return categorical
    

number_of_classes = 2
classes = one_hot_encoding(has_signal, number_of_classes)
classes = xarray.DataArray(data=classes,
                 dims=['sample_index', 'label'],
                 coords={'label':['noise','noise+signal']})
                              
validation = xarray.DataArray(data=is_for_validation,
                              dims=['sample_index'])

#number_of_sets = 2
#set_flag = one_hot_encoding(is_for_validation, number_of_sets)
#set_flag = xarray.DataArray(data=set_flag,
#                 dims=['sample_index', 'sets'],
#                 coords={'sets':['train','validation']})

dataset = xarray.Dataset(data_vars={'images':images, 'classes':classes, 'is_for_validation':validation})

dataset.to_netcdf('/storage/users/Muciaccia/images.netCDF4', format='NETCDF4')

# TODO risolvere il problema delle strane righe orizzontali ricorrenti


# TODO md5sum **/*
# TODO md5sum **/*.SFDB09 >> md5.txt
# TODO shiftare temporalmente dati Virgo (ed aggiungere i due file H finali)
# TODO controllare a mano spettri di virgo e fare scatterplot di tutti e tre per la classificazione
# TODO chiedere a Ornella codice e dati grezzi per generare le SFDB





exit()

# TODO mettere scala di frequenza sulle y e scala di GPS_time sulle x e valore di intensità/ampiezza nella colorbar
pyplot.figure(figsize=[10,20]) # TODO capire perché non scala
fig = pyplot.imshow(log_image, origin="lower", interpolation="none") # aspect='auto' or 'equal'
#pyplot.axis('off')
extent = fig.get_window_extent().transformed(pyplot.gcf().dpi_scale_trans.inverted()) # TODO brutto hack per non fargli mettere i bordi (che dovrebbe essere invece una cosa semplicissima
pyplot.savefig('/storage/users/Muciaccia/media/esempio_RGB.jpg', dpi=300, bbox_inches=extent) # oppure tiff. evitare png perché introduce parecchie corruzioni. svg purtroppo qui non supporta la non interpolazione
pyplot.close()
#pyplot.show()

exit()

prova_bianconero = dataset.mean(dim='detector')
prova_bianconero.spectrogram[0:256,0:128].plot() # TODO dentro il singolo elemento si può fare la selezione così come si fa su numpy
# frequenza da 80.0001 a 80.0312 Hz
pyplot.show()

b = dataset.spectrogram.expand_dims(dim='sample_index', axis=0)

b = good_dataset.spectrogram
b = b.expand_dims(dim=['frequency_block_index','time_block_index'], axis=[0,2]) # TODO rimuovere specifica sugli assi (tanto importa solo il nome)


# TODO VSR4
# provare VPN
# /storage/pss/virgo/sfdb/VSR4/

rows, columns, channels = good_dataset.spectrogram.shape
joined_RGB_images
#dataset.expand_dims(dim, axis=None)
#dataset.filter_by_attrs
#dataset.groupby
#dataset.groupby_bins
#dataset.reindex
#dataset.reorder_levels
#dataset.shift
#dataset.swap_dims
#dataset.stack
#dataset.unstack
#MultiIndex








# TODO nota su un BUG: numpy.zeros(5).shape è uguale a (5,) invece che a 5 o [5] perché così si può sempre chiamare axis=0 ad esempio in numpy.concatenate? non c'è un modo più intelligente per farlo?
# TODO BUG: non si può fare numpy.concatenate o numpy.stack tra un vettore (array) e una matrice. serve per forza aumentare fittizziamente le dimensioni del vettore in modo da farlo diventare una matrice Nx1

exit()


# TODO fare tassellazione a pezzi di 100

time_ticks_required = 128 # TODO valore semplice provvisorio
# fft_required = 100


# TODO migliorarlo (e farlo sui tempi invece che sugli indici)
good_indices = []
good_index = numpy.argmax(goodness_indicator)
while goodness_indicator[good_index] >= 1 - nan_tolerance:
    good_indices.append(good_index)
    goodness_indicator[good_index-50:good_index+50] = 0
    good_index = numpy.argmax(goodness_indicator)
#good_indices = numpy.array(good_indexes)

#good_times = L_dataset.GPS_time[good_indices]


# trovare il modo per fare multiple slicing su pandas e numpy

#minima = good_indexes - 50
#maxima = good_indexes + 50

# modo molto poco elegante
slice_list = []
for good_index in good_indices:
    slice_list.append(slice(good_index-50, good_index+50))

#target[good_indices]

good_indices = numpy.r_[tuple(slice_list)]
good_times = L_dataset.GPS_time.values[good_indices]

cutted_L_dataset = L_dataset.sel(GPS_time = good_times)


image_frequency_window = 0.1 # Hz
frequency_interval = 128 # TODO hardcoded
frequency_cuts = frequency_interval/image_frequency_window

time_cuts = len(slice_list) # rinominare good_indices

number_of_images = frequency_cuts * time_cuts

joined_images = numpy.squeeze(cutted_L_dataset.spectrogram.values)

# TODO farlo direttamente con xarray
splitted_images = joined_images.reshape(-1, int(frequency_cuts), 100, int(time_cuts))

splitted_images = numpy.transpose(splitted_images, axes=[1,3,0,2])

splitted_images = splitted_images.reshape(int(number_of_images), -1, 100)

# TODO BUG: dovrebbe essere semplice creare una lista di slices per l'ndicizzazione multipla. dovrebbe essere una cosa tipo numpy.slice(numpy.transpose(minima, maxima))

# in questo modo, con 2 soli indici semibuoni, si riescono a creare 2560 immagini bianconero 256x100 pixel

# L è discreto. H fa ancora veramente schifo

# TODO organizzare le immagini con xarray e salvarle in un netCDF4 per utilizzarle in futuro. savare l'informazione sulla frequenza e sull'intervallo temporale utilizzato

# xarray resample in frequenza?

# # if distance from a good index is less than 50
# is_sufficiently_dense = numpy.any(...)

# L_dataset.expand_dims

def display_raw_image(image):
    pyplot.figure(figsize=[10,25.6])
    pyplot.imshow(image, cmap='gray', norm=matplotlib.colors.LogNorm(vmax=1e-2, vmin=1e-12), aspect='auto', origin="lower", interpolation="none") # TODO controllare i livelli di minimo e massimo del rumore
#pyplot.colorbar()
    #pyplot.savefig('/home/federico/Desktop/possibile_segnale.tif')
    pyplot.show()
    
# TODO le strisce bianche rimangono bianche anche invertendo la colormap
# TODO invertendo la colormap però l'immagine diventa globalmente più scura, segno che la scala di grigio non è correttamente centrata
# TODO le immagini pari hanno una struttura di linee bianche, le immagini dispari hanno l'altra

#expanded_image = numpy.repeat(numpy.repeat(image, 5, axis=0), 5, axis=1)

display_raw_image(splitted_images[100])
pyplot.savefig('/home/federico/Desktop/possibile_segnale.tif')

import skimage.io
skimage.io.imsave('/home/federico/Desktop/possibile_segnale.tif', splitted_images[100])




#N
#conv
#N-1
#pool
#(N-1)/2
#
#255
#127
#63
#31
#15
#7
#3
#1*1*2 -> 2 -> 1
#
#si potrebbe mettere una convoluzione extra inziale dentro il blocco in modo da passare da 256*128 a 255*127 e poi fare dunque la sequenza corretta
#
#N
#zeros
#N+1
#conv
#N
#pool
#N/2
#
#256
#128
#64
#32
#16
#8
#4
#2
#1
#MA con immagini piccole non ha alcun senso fare lo zero_padding

def create_peakmap(image):
    pass


numpy.apply_along_axis


# TODO fare istogramma valori per stabilire il grigio centrale dell'immagine

# TODO magari non fare il whitening per preservare la struttura orizzontale di alcune linee risonanti

# TODO facendo un istogramma logaritmico si vede che i valori del rumore non sono gaussiani, quindi è giusto che le due colormap 'gray' e 'gray_r' non diano immagini uguali
# TODO non capisco però perché i valori del logaritmo risultano negativi e con estremi diversi da quelli espressi in precedenza (sono valori orientativamente doppi)
#pyplot.hist(numpy.log(image.flatten()+1e-20), bins=100, range=[-25, -5]);pyplot.show()


# in splitted_images[100] sembra esserci un transiente

# non capisco perché  alcune righe nette sembrano sparire nel tempo, come ad esempio in splitted_images[4]

# TODO controllare che lo spettrogramma sia effettivamente messo a zero dove i dati non sono science ready (altrimenti non si possono fare le immagini)




exit()



# TODO numpy.isin(element, test_elements) calculates `element in test_elements`, broadcasting over `element` only
# numpy.in1d
# numpy.lib.arraysetops

# TODO vedere se su un solo detector ci sono dati contigui per 5 giorni o quasi, in modo da poter cominciare a costruire le prime immagini monocromo

a = xarray.open_dataset('~/Desktop/RGB_dataset.netCDF4')

# TODO attenzione che ora con oggetti 3D non funziona più il plot

#RGB_images = RGB_dataset.spectrogram.values

# DA_FARE_OGGI:
# scrivere singoli file da 100 su disco, in modo da averli pronti e poi poterli caricare in parallelo senza sforzo
# output_folder su script Matlab
# off core computation e scrittura di singoli files senza concatenarli ed unlimited dimensions, anche per poi poterli leggere in parallelo (dask computing)
# far vedere codice Matlab a Cristiano per inserirlo in Snag. dividerlo in load_SFDB09 e convert_SFDB09, separando le funzioni
# numpy shape come tupla vs numpy.array tra i difetti
# numpy axis vs dimension. numeri vs stringhe coi nomi
# nuovi dati copiati a Ornella
# cominciare a scrivere capitolo tesi su data_loading_and_preprocessing.md (markdown). vedere estenzione gedit per conveersione in real time
# fare tutte le affiliazioni e gli account. consegnare richiesta tesi
# numpy or pandas or xarray categorical
# levare start_ISO_time dal dataset combinato
# eliminare la cartella /storage su wn100 perché è una copia? (attenzione che il calcolo andrà fatto su GPU)
# capire perché i dati sfdb di Ornella sono meglio organizzati di quelli che ho io
# gli altri dati di marzo stanno al CNAF
# CNAF, cartella Ornella, wn100/storage, wn1/storage, wn1/data
# vedere data alla quale è cambiata o cambierà la calibrazione da C00 a C01 
# il limite a 128 Hz deriva dall'aver preso FFT a 8192 s o dal subsampling time?
# cancellare i file .mat una volta creati i Dataset con xarray
# trovare file con elenco dei file fino a marzo
# prendere adattatore
# scrivere a Cristiano per Matlab
# facendo gli spettrogrammi (incoerenti) si perde l'informazione sulla fase delle onde (forse è meglio fare la rete neurale audio multiscala che proponeva Uncini)
# valutare di mettere seconda ssd economica per tenere i dati in locale (fare un calcolo di quandi mesi ci sono in un run della macchina)
# /data è condiviso da wn 1 2 3
# /storage è condiviso anche con wn 100


#unlimited_dims : sequence of str, optional
#    Dimension(s) that should be serialized as unlimited dimensions.
#    By default, no dimensions are treated as unlimited dimensions.

# netCDF, which is a binary file format for self-described datasets that originated in the geosciences. xarray is based on the netCDF data model, so netCDF files on disk directly correspond to Dataset objects
# NetCDF is supported on almost all platforms, and parsers exist for the vast majority of scientific programming languages.
# Recent versions of netCDF are based on the even more widely used HDF5 file-format.
# Reading and writing netCDF files with xarray requires scipy or the netCDF4-Python library to be installed
# ds.to_netcdf('saved_on_disk.nc')
# By default, the file is saved as netCDF4 (assuming netCDF4-Python is installed)
# ds_disk = xr.open_dataset('saved_on_disk.nc')
# Data is always loaded lazily from netCDF files. You can manipulate, slice and subset Dataset and DataArray objects, and no array values are loaded into memory until you try to perform some sort of actual computation.
# It is important to note that when you modify values of a Dataset, even one linked to files on disk, only the in-memory copy you are manipulating in xarray is modified: the original file on disk is never touched.
# xarray’s lazy loading of remote or on-disk datasets is often but not always desirable. Before performing computationally intense operations, it is often a good idea to load a Dataset (or DataArray) entirely into memory by invoking the load() method.
# Datasets have a close() method to close the associated netCDF file. However, it’s often cleaner to use a with statement:
# Although xarray provides reasonable support for incremental reads of files on disk, it does not support incremental writes, which can be a useful strategy for dealing with datasets too big to fit into memory. Instead, xarray integrates with dask.array (see Out of core computation with dask), which provides a fully featured engine for streaming computation.
# format='netCDF4' and either engine='netcdf4' or engine='h5netcdf'
# Version 0.5 includes support for manipulating datasets that don’t fit into memory with dask. If you have dask installed, you can open multiple files simultaneously using open_mfdataset():
# xr.open_mfdataset('my/files/*.nc')
# This function automatically concatenates and merges multiple files into a single xarray dataset. It is the recommended way to open multiple files with xarray.
#from glob import glob
#import xarray as xr
#
#def read_netcdfs(files, dim):
#    # glob expands paths with * to a list of files, like the unix shell
#    paths = sorted(glob(files))
#    datasets = [xr.open_dataset(p) for p in paths]
#    combined = xr.concat(dataset, dim)
#    return combined
#
#combined = read_netcdfs('/all/my/files/*.nc', dim='time')
#
#This function will work in many cases, but it’s not very robust. First, it never closes files, which means it will fail one you need to load more than a few thousands file. Second, it assumes that you want all the data from each file and that it can all fit into memory. In many situations, you only need a small subset or an aggregated summary of the data from each file.
#
#Here’s a slightly more sophisticated example of how to remedy these deficiencies:
#
#def read_netcdfs(files, dim, transform_func=None):
#    def process_one_path(path):
#        # use a context manager, to ensure the file gets closed after use
#        with xr.open_dataset(path) as ds:
#            # transform_func should do some sort of selection or
#            # aggregation
#            if transform_func is not None:
#                ds = transform_func(ds)
#            # load all data from the transformed dataset, to ensure we can
#            # use it after closing each original file
#            ds.load()
#            return ds
#
#    paths = sorted(glob(files))
#    datasets = [process_one_path(p) for p in paths]
#    combined = xr.concat(datasets, dim)
#    return combined
#
# here we suppose we only care about the combined mean of each file;
# you might also use indexing operations like .sel to subset datasets
#combined = read_netcdfs('/all/my/files/*.nc', dim='time', transform_func=lambda ds: ds.mean())




#%%%%%%%%%%%%%%%%%%%%%%%%


print('detectors: \t LIGO Hanford, LIGO Livingston')
print('observing run: \t O2') # most recent data
print('calibration: \t C00') # first calibration, then we will use C01
print('SFDB band: \t "256" (up to 128 Hz)') # TODO la banda "256" è perché il subsampling time è 1/256? (mentre la frequneza massima è 128)
# possible SFDB bands: 256, 512, 1024, 2048

data_dir = '/storage'

LIGO_Hanford_data_dir = '/storage/pss/ligo_h/sfdb/O2/128/'
LIGO_Livingston_data_dir = '/storage/pss/ligo_l/sfdb/O2/128/'
# TODO i dati in /storage sono delle copie di /data

#files = os.listdir(LIGO_Livingston_data_dir) # restituisce solo i nomi dei file/cartelle e non il loro percorso

#import fnmatch

#os.path # TODO

#mat_files = fnmatch.filter(files, '*.mat')

import glob

LIGO_Livingston_mat_files = glob.glob(LIGO_Livingston_data_dir + '*.mat') # non funziona con la data_dir generale

# TODO ordinare la lista di file in ordine cronologico
LIGO_Livingston_mat_files = sorted(LIGO_Livingston_mat_files)


#%%%%%%%%%%%%%%%%%%%%%%%%





#%%%%%%%%%%%%%%%%%%%%%%%%

selected_power_spectrum = []
fft_index = []
science_ready = []
detector = []
for mat_file in LIGO_Livingston_mat_files:
    a,selected_frequencies,c,d,detector = process_file(mat_file)
    selected_power_spectrum.append(a)
    fft_index.append(c)
    science_ready.append(d)
    # TODO molto poco elegante e conciso (provare a farlo con le strutture (e in maniera vettoriale))
selected_power_spectrum = numpy.concatenate(selected_power_spectrum)
fft_index = numpy.concatenate(fft_index) # TODO dato che l'indice ricomincia periodicamente, magari usare semplicemente la data e il tempo di acquisizione per garantire il buo ordinamento dei dati (e l'eventuale corretto conteggio di interruzioni temporali)
science_ready = numpy.concatenate(science_ready)

# TODO fare una funzione process_directory() che ritorni solo questo, le frequenze (y) e le date/tempi (coordinate temporali numeriche: GPS o UTC) (x)
# usare xarray in modo da poter fare slices agevolmente: data.frequency[80:120] e anche data.time[2016-12-01:2017-02-9]
# usare l'indice temporale come una TimeSeries
# gps_times[science_ready]
clean_selected_power_spectrum = selected_power_spectrum[science_ready]

# TODO mettere la possibilità di specificare finestra temporale e finestra di frequenze direttamente nella funzione process_file()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



# crea l'immagine per soltanto 1 Hz di banda
# dunque immagine 8192x100
# 2^13 = 8192
image = selected_power_spectrum #[:,:8192]
# oppure image = clean_selected_power_spectrum se si vuole fare collassare vicino tutti i valori interessanti
# oppure image = clean_power_spectrum per verificare il costante allineamento nel tempo delle riche principali, per essere così sicuri che non ci siano shift macroscopici (si vedono bene le due righe a 60 Hz e a 100 Hz, quindi non c'è scift apprezzabile nei dati)
# TODO cmap gray_r
pyplot.figure(figsize=[10,10])
pyplot.imshow(image.T, cmap='gray', norm=matplotlib.colors.LogNorm(vmax=1e-2, vmin=1e-12), aspect='auto', origin="lower", interpolation="none")
#pyplot.colorbar() # TODO far ricomparire i numeri a destra della colorbar
# TODO appena possibile ripristinare il rapporto naturale coi pixel
# TODO mettere frequnenze sulle x e fft_index sulle y (e poi trasporre l'immagine)
# TODO pyplot.imshow(image, extent=extent, interpolation="nearest",origin="lower")
# TODO capire perché con le frequenze sulle ascisse si riesce a vedere la riga a 100 Hx, mentre con le frequenze sulle ordinate non si riescee a vederla (attenzione! la visibilità della riga sembra anche essere funzione della dimensione della figura, dunque forse ci sono problemi con l'elaborazione dei toni di grigio a schermo)

# TODO scrivere funzione per salvare le immagini su disco (come immagini tiff e non come matrici)(stare attenti al verso corretto delle immagini)

# TODO poi da concatenare con numpy


######################


# dim = 1 5 è concettualmente sbagliato perché assume come riferimento R^2
# TODO dunque forse anche il sistema di numpy è sbagliato: numpy.arange(5).shape deve essere 5 e non (5,)
# numpy.arange(1).shape è (1,)
# numpy.array(4).shape è ()

# TODO l'assegnazione con `=` in python è molto pericolosa, perché alle volte è come l'uguaglianza simbolica, che modifica tutto a catena 

# NOTA: già dalla stessa dimensione dei file .mat compressi si può capire che i dati di LIGO Hanford sono sostanzialmente tutti zero nel periodo analizzato
# i dati di LIGO Livingston sembrano invece poter dare qualche speranza nei seguenti files, che tra l'altro sono contigui nel tempo:
# L1:GDS-CALIB_STRAIN_20161130_002935.SFDB09.mat
# L1:GDS-CALIB_STRAIN_20161204_181615.SFDB09.mat
# L1:GDS-CALIB_STRAIN_20161209_120255.SFDB09.mat
# L1:GDS-CALIB_STRAIN_20161214_054935.SFDB09.mat
# L1:GDS-CALIB_STRAIN_20161218_233615.SFDB09.mat
# NOTA: dal nome file, ogni file sembra contenere circa 5 giorni di dati (TODO controllare se è vero facendo il calcolo)

# TODO vedere se il motivo per cui il programma è lentissimo a scrivere su disco è la compressione

# TODO vedere cross spectral density

# TODO Snag sul server

# so we need a working Matlab installation, the Matlab Python engine and the Snag package installed
import matlab.engine
# start Matlab Python engine
eng = matlab.engine.start_matlab()



prova_path = './dati_di_prova/H1:GDS-CALIB_STRAIN_20161130_002935.SFDB09'


file_path = altro_path


prova_paths = ['./dati_di_prova/H1:GDS-CALIB_STRAIN_20161130_002935.SFDB09', './dati_di_prova/altri/L1:GDS-CALIB_STRAIN_20161228_110935.SFDB09']



# TODO
data_dir = './dati_di_prova/'
file_paths = os.listdir(data_dir)
# TODO all_files_in_dir_with_paths
os.chdir(data_dir) # TODO workaround
# TODO filtrare sui file con estensione corretta


# H semibuono, L no


# pyhton wrapper for the Matlab function
def read_sfdb_file(file_path):
    structure = eng.read_SFDB(file_path)
    header = structure['header'] # a dictionary
    # convert from matlab.double and then remove the useless extra dimension
    periodogram = numpy.array(structure['periodogram']).reshape(-1)
    autoregressive_spectrum = numpy.array(structure['autoregressive_spectrum']).reshape(-1)
    data = numpy.array(structure['data']).reshape(-1)
    return header, periodogram, autoregressive_spectrum, data # piahead tps sps sft



# vectorize the data reading
# TODO indicare direttamente la directory
def read_sfdb_files(file_paths):
    headers = []
    periodograms = []
    autoregressive_spectra = []
    data_sequence = []
    for file_path in file_paths:
        header, periodogram, autoregressive_spectrum, data = read_sfdb_file(file_path)
        headers.append(header)
        periodograms.append(periodogram)
        autoregressive_spectra.append(autoregressive_spectrum)
        data_sequence.append(data)
        # TODO convertire in array numpy
    return headers, periodograms, autoregressive_spectra, data_sequence



a['periodogram'].shape

for i in range(5):
    print(i)
    l = []
    a,b,c,d = read_sfdb_file(prova_path)
    print(d)
    l.append(d[:])
    print(l)



# stop Matlab Pyhton engine
eng.quit()




# 632 fft totali?

# TODO il codice di lettura di Pia non è parallelo sulle varie FFT

clean_data = numpy.array(list(zip(images,classes,amps,validation)), dtype='(98,82,1)float32, float32, float32, int8')
clean_data.dtype.names = ('image','class','amplitude','validation')
# save the data on disk
numpy.save('./data/clean_data.npy', clean_data)

# TODO wrapper from pure C++ instead of Matlab (Pia_read_block_09)

# dati copiati a Roma su virgo-wn100.roma1.infn.it/storage
# gli originali sono al CNAF a Bologna su virgo-wn1.roma1.infn.it/data

# pss = Periodic Source Search

# time_FFT = 8192 seconds
# 1 day = 86400 seconds
# FFT interlacciate, dunque ci sono circa 21 FFT interlacciate al giorno


detectors = ['LIGO Hanford','LIGO Livingston']
observing_run = 'O2' # most recent data
calibration = 'C00' # first calibration, then we will use C01
sfdb_band = 256 # or 512, 1024, 2048
min_frequency = 10 # TODO controllare
max_frequency = 128 # TODO controllare

print('detectors:', '\t', '{}, {}'.format(*detectors))
print('observing run:', '\t', observing_run)
print('calibration:', '\t', calibration)
print('SFDB band:', '\t', '"{}"'.format(sfdb_band), '(from {} Hz to {} Hz)'.format(min_frequency, max_frequency))

# every single SFDB09 file includes 100 FFT
number_of_FFT = 100

include_time_data = True

number_of_figures = 20 # TODO




# putroppo usare Matlab è qui necessario perché le funzioni per la lettura dei file preaggregati sono in una libreria scritta solo in Matlab
# TODO vedere eventuale versione in C++


# TODO assume there are no subfolders (only files)
LIGO_Hanford_files = os.listdir('/storage/pss/ligo_h/sfdb/O2/128/')
LIGO_Livingston_files = os.listdir('/storage/pss/ligo_l/sfdb/O2/128/')

fig, [ax1, ax2, ax3] = pyplot.subplots(3, sharex=True)
fig.suptitle('time:{}'.format(date), size=16)
ax1.set_title('normalized power FFT (blue), autoregressive mean (magenta), periodogram (black)')

percorso_file_di_prova = 'H1:GDS-CALIB_STRAIN_20161130_002935.SFDB09'


mjd_time # modified julian day
gps_time
utc_time


file_identifier = eng.fopen(path)
eng.pia_read_block_09(file_identifier)




