
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


import numpy
import dask
import xarray
import astropy.time
import matplotlib
matplotlib.use('SVG') # per poter girare lo script pure in remoto sul server, dove non c'è il server X
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
# TODO non viene fatta bene la concatenzaione lungo GPS_time. TODO capire perché
#dataset = xarray.open_mfdataset('/storage/users/Muciaccia/netCDF4/O2/C01/128Hz/*.netCDF4')#, chunks={'GPS_time': 100}) # hardcoded # TODO non legge le sottocartelle


H_dataset = xarray.open_mfdataset('/storage/users/Muciaccia/netCDF4/O2/C01/128Hz/LIGO Hanford*.netCDF4')
L_dataset = xarray.open_mfdataset('/storage/users/Muciaccia/netCDF4/O2/C01/128Hz/LIGO Livingston*.netCDF4')
# chunksize scelto automaticamente (quando si caricano file spezzettati)

# TODO fake Virgo dataset
#V_dataset = xarray.open_mfdataset('/storage/users/Muciaccia/fake_Virgo_dataset.netCDF4', chunks={'GPS_time': 100}) # TODO qui bisogna invece specificare il chunks_value manualmente
V_dataset = xarray.open_mfdataset('/storage/users/Muciaccia/netCDF4/O2/C01/128Hz/Virgo*.netCDF4')

# VSR4 shifted in time and amplitude
time_lenght = len(V_dataset.GPS_time)
start_time_index = 650#2500000 #0
new_times = H_dataset.GPS_time.isel(drop=True, GPS_time=slice(start_time_index,start_time_index+time_lenght)) # TODO dataset.isel({'GPS_time':slice(...), 'frequency':slice(...)})
# TODO artificially shift the VSR4 starting time
V_dataset.update({'GPS_time':new_times}) # V_dataset['GPS_time'] = new_times
# TODO artificially decrease the VSR4 amplitude (better sensitivity)
V_dataset['spectrogram'] = V_dataset.spectrogram * numpy.exp(-4)

dataset = xarray.concat([H_dataset,L_dataset,V_dataset], dim='detector')

# TODO in xarray.open_mfdataset attributes from the first dataset file are used for the combined dataset



fig, [raw, whitened] = pyplot.subplots(nrows=2, ncols=1, figsize=[10,10])
raw.semilogy(dataset.frequency, dataset.spectrogram[:,0,0])
raw.set_ylabel('strain') # TODO CHECK # TODO mettere unità di misura
raw.set_xlabel('frequency [Hz]')
whitened.semilogy(dataset.frequency, dataset.whitened_spectrogram[:,0,0])
whitened.set_ylabel('whitened strain') # TODO CHECK
whitened.set_xlabel('frequency [Hz]')
raw.set_title('comparison between raw and whitened spectra \n', size=16) # TODO 3 hack
pyplot.savefig('/storage/users/Muciaccia/media/whitening.jpg')
pyplot.close()

# TODO mettere i due istogrammi verticalmente a lato dei due spettri (come fannpo gli astrofisici per il grafico dei residui)
# TODO non serve a molto, dato che poi verrà fatto un plot simile coi tre colori/canali/detector. serve solo a far vedere che la distribuzione non viene molto modificata

numpy.log(dataset.spectrogram[0:256,0:128,0]).plot.hist(bins=100, range=[-20, 0])
pyplot.show()

numpy.log(dataset.whitened_spectrogram[0:256,0:128,0]).plot.hist(bins=100, range=[-10, 10])
pyplot.show()



#a.globally_science_ready[a.globally_science_ready.values == True]

# TODO eliminare attributo start_ISO_time
#globally_selected_continuous_dataset = a.where(a.globally_science_ready == True, drop=True)

# questo tipo di conti richiedono praticamente zero spazio in memoria :)

#locally_selected_continuous_dataset = a.where(a.locally_science_ready.any(dim='detector'), drop=True)

#locally_selected_dataset = a.where(a.locally_science_ready.any(dim='detector'))

#L_dataset = dataset.where(dataset.detector == 'LIGO Livingston', drop=True)

#H_dataset = dataset.where(dataset.detector == 'LIGO Hanford', drop=True)

#selected_L_dataset = L_dataset.where(L_dataset.locally_science_ready == True)

second = 1
minute = 60 * second
hour = 60 * minute
day = 24 * hour
week = 7 * day

default_time_scale = week # TODO

def time_pixels(time_interval):
    # time_interval is in seconds
    time_delta = dataset.FFT_lenght
    return 2*int(numpy.ceil(time_interval/time_delta))
    # approssimazione per eccesso all'intero più vicino
    # il fattore 2 serve a tenere in conto del fatto che le FFT sono interallacciate

# magari fare il calcolo con una convoluzione 1D e mettere una soglia sul valore

def time_stability(data, time_interval = default_time_scale):
    #time_interval = 128
    number_of_time_ticks = time_pixels(time_interval)
    kernel = numpy.ones(number_of_time_ticks)
    target = data
    # TODO questa convoluzione è una sorta di running average (?)
    stability_indicator = numpy.convolve(kernel, target, mode='valid')/number_of_time_ticks # mode='same'
    return stability_indicator.astype(numpy.float32) # to better handle the NaN values
# TODO apply_along_dimension with xarray

#kernel = numpy.ones(time_ticks_required)
#target = H_dataset.locally_science_ready.values.flatten()
#goodness_indicator = numpy.convolve(kernel, target, mode='same')/time_ticks_required
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

#H_time_stability = five_day_time_stability(H_dataset.locally_science_ready.values.flatten())
H_time_stability = time_stability(dataset.locally_science_ready.sel(detector='LIGO Hanford').values.flatten())
L_time_stability = time_stability(dataset.locally_science_ready.sel(detector='LIGO Livingston').values.flatten())
V_time_stability = time_stability(dataset.locally_science_ready.sel(detector='Virgo').values.flatten())

# find coincidences
#dataset.locally_science_ready.notnull()
globally_science_ready = dataset.locally_science_ready.all(dim='detector')
# TODO BUG: numpy.nan conta come True

combined_time_stability = time_stability(globally_science_ready.values)

pyplot.figure(figsize=[15,10])
pyplot.plot(H_time_stability, label='LIGO Hanford', color='#ff2d33') # plot per vedere come evolve nel tempo la bontà dei dati (loro densità temporale locale in funzione del tempo)
pyplot.plot(L_time_stability, label='LIGO Livingston', color='#208033')
pyplot.plot(V_time_stability, label='Virgo', color='#3366ff')
pyplot.plot(combined_time_stability, label='all detectors together', color='#404040')
#pyplot.axhline(y=acceptable_percentage, color='green', label='acceptable level ({}%)'.format(int(acceptable_percentage*100)))
pyplot.title('{} {} data stability on 1-week timescale'.format(dataset.observing_run, dataset.calibration))
pyplot.xlabel('time') # GPS time
pyplot.xticks(time_ticks, month_labels)
pyplot.ylabel('density of FFTs on 1-week timescale') # TODO vedere nome migliore
#pyplot.xlim([0,5*month]) # TODO
pyplot.ylim([0,1])
pyplot.legend(loc='upper right', frameon=False)
#pyplot.show()
pyplot.savefig('/storage/users/Muciaccia/media/time_stability.svg', dpi=300)
pyplot.close()
# TODO sistemare i colori
# TODO mettere legenda con due label con detector e caratteristiche del run e tick temporali coi mesi
# TODO mettere legenda in RGB, così come tutte le 3+1 (CMY + W) possibili combinazioni di detector simultanei


combined_time_stability[numpy.isnan(combined_time_stability)] = 0

# TODO tassellazione provvisoria, valida in regime di bassa densità

#nan_tolerance = 0.3 # 30%
#acceptable_percentage = 1 - nan_tolerance





#minimum_acceptable_combined_density = 0.3
# TODO migliorarlo (e farlo sui tempi invece che sugli indici)
good_slices = []
good_index = numpy.argmax(combined_time_stability)
#while combined_time_stability[good_index] >= minimum_acceptable_combined_density:
while len(good_slices) < 2: # TODO farlo out-of-memory per non avere limitazioni sul campione
    # TODO attenzione che ci sono spesso sovrapposizioni
    half_interval = int(time_pixels(default_time_scale)/2) # TODO assicurarsi del giusto arrotondamento
    good_slice = slice(good_index-half_interval, good_index+half_interval)
    good_slices.append(good_slice)
    combined_time_stability[good_slice] = 0
    good_index = numpy.argmax(combined_time_stability)

# TODO che succede con le sovrapposizioni? si ripete/duplica parte del dataset?
good_dataset = dataset.isel(GPS_time = numpy.r_[tuple(good_slices)]) # TODO cercare un modo più elegante e comprensibile
# TODO valutare se concatenare dei numpy.arange
#mgrid = nd_grid(sparse=False)
#ogrid = nd_grid(sparse=True)
#xs, ys = numpy.ogrid[0:5,0:5]



# good_index = numpy.argmax(combined_time_stability)
# good_slice = slice(good_index-64, good_index+64) # TODO hardcoded
# #dataset.GPS_time.isel(good_slice, drop=True)
# #dataset.isel({'GPS_time':good_slice})
# # .isel .isel_points # TODO
# #good_times = dataset.GPS_time.values[good_slice]
# # isel
# # TODO view VS copy (in numpy)
# #good_dataset = dataset.sel(GPS_time = good_times) # TODO ???
# good_dataset = dataset.isel(GPS_time = good_slice) # TODO ???

# si possono combinare anche le condizioni:
# H_dataset.isel(GPS_time = slice(0,128), frequency = slice(0,256))

## 327680 pixels along frequency axis
## frequencies are from 80 Hz to 120 Hz
#
#frequency_pixels_in_the_whole_spectrogram = len(cutted_dataset.frequency)
#
#number_of_frequency_divisions = frequency_pixels_in_the_whole_spectrogram / image_frequency_pixels # TODO round the float value
#
##image_frequency_interval = 0.4 # Hz
##image_frequency_interval = 0.1 # Hz
#image_frequency_interval = total_frequency_interval / number_of_frequency_divisions

frequency_resolution = 1/dataset.FFT_lenght # TODO vedere se c'è il fattore 2 per le FFT interallacciate (non capisco come faccia a fare lo stesso risultato derivante del codice commentato qui sopra) (in effetti ogni pixel rappresenta una FFT. il fatto che ci siano le FFT interallacciate vuol dire che ci saranno il doppio dei pixel dantro un dato intervallo di frequenze, dunque ci dovrebbe comunque essere un fattore 2)
image_frequency_pixels = 256 # TODO 1024 # TODO hardcoded
image_frequency_interval = frequency_resolution * image_frequency_pixels # TODO ??? fattore 2? # TODO facendo il plot direttamente con xarray il valore risulta giusto (fino alla quarta cifra)
total_frequency_interval = 120 - 80 # TODO hardcoded
frequency_divisions = int(total_frequency_interval / image_frequency_interval) # TODO = int(len(dataset.frequency)/image_frequency_pixels)

time_divisions = len(good_slices) #int(1)

number_of_images = int(frequency_divisions * time_divisions) # TODO attenzione agli errori di troncamento

image_time_pixels = time_pixels(default_time_scale)

channels = 3


# TODO PARTE LENTISSIMA E COSTRETTA DALL'AMMONTARE DI MEMORIA


# TODO la lista è ancora un array numpy in memoria. provare dataset.groupby()
#first_image = numpy.split(dataset.whitened_spectrogram,number_of_frequency_divisions,axis=dataset.whitened_spectrogram.get_axis_num('frequency'))[0]
#pyplot.figure(figsize=[5,10])
#numpy.log(first_image).plot(vmin=-10, vmax=5)
#pyplot.show()

#joined_RGB_images = good_dataset.spectrogram.values
big_RGB_image = dask.array.from_array(good_dataset.whitened_spectrogram, chunks=128) # TODO mettere chunks automatici o ereditati da xarray
# con numpy si riempe subito quasi tutta la memoria
# TODO farlo direttamente con xarray invece che con dask, in modo da mantenere i valori di frequency e GPS_time (comodo soprattutto nei plot)

# TODO rivedere nomi poco chiari
# frequency = rows*height
# time = columns*width
splitted_images = big_RGB_image.reshape(frequency_divisions, image_frequency_pixels, time_divisions, image_time_pixels, channels) # rows, height, columns, width, channels
splitted_images = splitted_images.transpose(0,2,1,3,4) # rows, columns, height, width, channels
# number_of_images = rows*columns
RGB_images = splitted_images.reshape(number_of_images, image_frequency_pixels, image_time_pixels, channels)

# create many little images by tassellation of the big image
# 1) frequency, time, channels
# 2) rows*height, columns*width, channels
# 3) rows, height, columns, width, channels
# 4) rows, columns, height, width, channels
# 5) rows*columns, height, width, channels
# 6) number_of_images, height, width, channels

#samples, height, width, channels = imgs.shape
## samples is now equal to rows * cols
## let's rearrange things to create a grid of images
#mosaic = imgs.reshape(rows, cols, height, width, channels)
## permutation of some indices
#mosaic = numpy.transpose(mosaic, axes=(0,2,1,3,4))
#mosaic = mosaic.reshape(rows*height, cols*width, channels)

# # esempio che funziona:
# frequency = 20
# time = 6
# channels = 1
# m = numpy.arange(120).reshape(frequency, time, channels)
# print(m.squeeze())
# # frequency = rows*height
# # time = columns*width
# rows = 4
# height = 5
# columns = 2
# width = 3
# m1 = m.reshape(rows, height, columns, width, channels)
# m2 = m1.transpose(axes=[0,2,1,3,4]) # rows, columns, height, width, channels
# print(m2.squeeze())
# # number_of_images = rows*columns
# m3 = m2.reshape(rows*columns, height, width, channels)
# print(m3.squeeze())


#image = RGB_images[0]

# TODO analizzare tutti gli script con un profiler, in modo da vedere dove sono i colli di bottiglia e cosa può essere ulteriormente velocizzato e parallelizzato magari scrivendolo diversamente

# TODO valutare se spostare una parte di codice su TensorFlow, usando le parallelizzazioni su GPU e i data generators

# TODO farlo con xarray e salvare in .netCDF4 per poter fare computazione out-of-memory
#numpy.save('/storage/users/Muciaccia/background_RGB_images.npy', RGB_images)

background_RGB_images = xarray.DataArray(data=RGB_images, dims=['image_index', 'height', 'width', 'channels'], name='images')

background_RGB_images.to_netcdf('/storage/users/Muciaccia/background_RGB_images.netCDF4', format='NETCDF4')

