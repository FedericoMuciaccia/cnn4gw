
import numpy
import xarray
import astropy.time
import matplotlib
from matplotlib import pyplot



#######################

# convert_mat_to_netCDF4(input_file_or_folder, output_folder)

# load_the_various_datasets_in_chunks_in_parallel
# preprocess_data
# save_the_big_dataset
# reload_it_in_chunks

#######################

# TODO adesso script create_images.py

# fino a fine febbraio: 1917 tempi, 502 any, 15 all

# TODO poi creare peakmap monocromo per comparare le performances col caso del rumore bianco gaussiano

# TODO
# PAY ATTENTION: GPS_time duration must here be the same for both detectors # TODO fatto a mano # TODO risolverlo con le concatenazioni coi NaN

# TODO load in chunks (out-of-memory computation) # TODO controllare se la lettura in chuncks viene fatta automaticamente con dataset grandi
# viene automaticamente fatta lla concatenazione sui tempi: comodissimo!
dataset = xarray.open_mfdataset('/storage/users/Muciaccia/netCDF4/O2/C01/128Hz/*.netCDF4')#, chunks={'GPS_time': 100}) # hardcoded # TODO non legge le sottocartelle

compat : {'identical', 'equals', 'broadcast_equals',
          'no_conflicts'}, optional
    String indicating how to compare variables of the same name for
    potential conflicts when merging:

    - 'broadcast_equals': all values must be equal when variables are
      broadcast against each other to ensure common dimensions.
    - 'equals': all values and dimensions must be the same.
    - 'identical': all values, dimensions and attributes must be the
      same.
    - 'no_conflicts': only values which are not null in both datasets
      must be equal. The returned dataset then contains the combination
      of all non-null values.


#a.globally_science_ready[a.globally_science_ready.values == True]

# TODO eliminare attributo start_ISO_time
#globally_selected_continuous_dataset = a.where(a.globally_science_ready == True, drop=True)

# questo tipo di conti richiedono praticamente zero spazio in memoria :)

#locally_selected_continuous_dataset = a.where(a.locally_science_ready.any(dim='detector'), drop=True)

#locally_selected_dataset = a.where(a.locally_science_ready.any(dim='detector'))

L_dataset = dataset.where(dataset.detector == 'LIGO Livingston', drop=True)

H_dataset = dataset.where(dataset.detector == 'LIGO Hanford', drop=True)

#selected_L_dataset = L_dataset.where(L_dataset.locally_science_ready == True)

time_delta = 8192 # TODO hardcoded
day = 60*60*24
required_time = 5*day
time_ticks_required = 2*int(numpy.ceil(required_time/time_delta)) # approssimazione per eccesso all'intero più vicino (ricordando che il fattore 2 serve a tenere in conto del fatto che le FFT sono interallacciate)
# TODO servono quindi 106 tempi. ma a questo punto direi che si potrebbe arrotondare a 100 per semplicità

# magari fare il calcolo con una convoluzione 1D e mettere una soglia sul valore

nan_tolerance = 0.3 # 30%

time_ticks_required = 100

kernel = numpy.ones(time_ticks_required)
target = H_dataset.locally_science_ready.values.flatten()
goodness_indicator = numpy.convolve(kernel, target, mode='same')/time_ticks_required
#pyplot.hist(goodness_indicator, bins=time_ticks_required, range=[0,1])

#is_sufficiently_dense = goodness_indicator >= 1 - nan_tolerance


# # TODO BUG: 5_days_stability_indicator non va bene come nome perché include un numero
# lambda_convolution = lambda target: numpy.convolve(kernel, target, mode='same')/100 # TODO hack terribile perché non si può applicare la convoluzione rispetto a un dato asse
# stability_indicator = numpy.apply_along_axis(lambda_convolution, axis=0, arr=dataset.locally_science_ready.values)
# 
# stability_indicator = xarray.DataArray(data=stability_indicator, 
#                                        dims=['GPS_time','detector'], 
#                                        coords=[gps_time_values, [detector]])
# 
# dataset.update({'stability_indicator': (['GPS_time','detector'], stability_indicator)})


month = 60*60*24*30 # in seconds (approximate value)
month = 2 * month/8192 # in interlaced fft
iso_time_ticks = ['2016-11-01', '2016-12-01', '2017-01-01', '2017-02-01', '2017-03-01']
iso_time_ticks = astropy.time.Time(val=iso_time_ticks, format='iso', scale='utc')
gps_time_ticks = iso_time_ticks.gps

# effettivamente si comincia da 2016-11-30 00:29:35.000
time_ticks = numpy.arange(6)*month
month_labels = ['Dec 2016', 'Jan 2017', 'Feb 2017', 'Mar 2017', 'Apr 2017', 'May 2017']


pyplot.figure(figsize=[15,10])
pyplot.plot(goodness_indicator, label='LIGO Hanford O2 C01') # plot per vedere come evolve nel tempo la bontà dei dati (loro densità temporale locale in funzione del tempo) # TODO hardcoded
pyplot.axhline(y=1-nan_tolerance, color='green', label='acceptable level')
pyplot.title("detector's 5-day time stability")
pyplot.xlabel('time') # GPS time
pyplot.xticks(time_ticks, month_labels)
pyplot.ylabel('5-day fft density') # TODO vedere nome migliore
pyplot.ylim([0,1])
pyplot.legend(loc='upper right', frameon=False)
pyplot.show()
# TODO mettere legenda con due label con detector e caratteristiche del run e tick temporali coi mesi


exit()


# TODO fare tassellazione a pezzi di 100

time_ticks_required = 100 # TODO valore semplice provvisorio
# fft_required = 100

# tassellazione provvisoria, valida in regime di bassa densità

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
frequency_interval = 128 # hardcoded
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




