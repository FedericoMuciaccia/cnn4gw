
import numpy
import xarray
import scipy.io
import matplotlib
from matplotlib import pyplot
import astropy.time
import glob

# %matplotlib inline

#%%%%%%%%%%%%%%%%%%%%%%






# TODO vedere modulo interno di python per l'elaborazione dei path

# file craeate_peakmap.m

# fare rumore vero con e senza soglia

#altro_path = './dati_di_prova/altri/L1:GDS-CALIB_STRAIN_20161228_110935.SFDB09'

file_path = altro_path = '/storage/pss/ligo_l/sfdb/O2/128/L1:GDS-CALIB_STRAIN_20161209_120255.SFDB09.mat'


# TODO scipy does not support v7.3 mat-files
# matlab --v7.3 files are hdf5 datasets
# import h5py 
# f = h5py.File('somefile.mat','r') 
# data = f.get('data/variable1') 
# data = np.array(data)

# periodogram = tps
# autoregressive_spectrum = sps
# data = sft

show_time_data = True # TODO plottare i dati nel tempo per vedere le sequenze temporali dove sono stati messi gli zeri e che poi rovinano le fft

# TODO t_FFT FFT_time_window

# TODO vedere pacchetto di LIGO GWpy

# TODO i valori della fft sono complessi, mentre i due spettri sono reali. vedere complex32 su GPU

# TODO Dataset.groupby(detector)

    ## unused attributes:
    # endianess
    # fft_interlaced # TODO
    # lenght_of_averaged_time_spectrum # TODO ?
    # mjd_time
    # number_of_flags # TODO
    # position
    # velocity
    # starting_fft_sample_index
    # subsampling_time
    # unilateral_number_of_samples
    # window_type

    # TODO vedere come creare direttamente array booleani in numpy (esempio: numpy.false(shape=(10,20)))
    # is_flagged = a['is_flagged'].astype(bool)
    
    # TODO iterare pure sulla dimensione dell'indice (usando xarray?)
    
    # TODO autoregressive_mean vs autoregressive_spectrum
    
    # risoluzione 8192/2 secondi per non poter mai uscire dal canale in frequenza
    
    # TODO flag di zeri nel tempo per il calcolo delle fft
    
    # TODO flag locked
    
    # TODO dilatare (raddoppiare) la scala temporale

        #from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
        #
        #fig = plt.figure(1)
        #ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])
        #
        #r, g, b = get_rgb() # r,g,b are 2-d images
        #ax.imshow_rgb(r, g, b,
        #              origin="lower", interpolation="nearest")
        
        
        # su Matlab gli scalari hanno size 1 1: 1 riga e 1 colonna. questo credo sia un errore concettuale derivante dal fatto di essere centrati sulle sole matrici, ovvero oggetti 2D


# Pia:
# numpy.sqrt(1 - number_of_zeros/unilateral_number_of_samples)
# number_of_flags con alcuni 0 e tutti gli altri -1
# grafici buoni ad occhio ma con differenza tra autoregressivo e periodogramma
# vettore unilateral_number_of_samples
# significato L1, GDS (?), CALIB_STRAIN (strain calibrato) nel nome file
# grandezza sull'asse y/z dei grafici (strain? 1/sqrt(Hz) densità spettrale)

# 1/sqrt(1-zeros/number) # TODO
# spettro unilatero = sqrt(2) * bilatero # TODO
# scaling factor arbitrario (attenzione al troncamento) (levarlo da tutti)

# nsamples è scalare (e dovrebbe essere uguale anche nell'ultimo file)
# flag = -1 (o 1?) per con interferometro lockato (dati anche non-science ma con interferometro locked)
# qui ignorare flag

# spettro autoregressivo poi usato per le peakmap, quindi il livello deve essere buono

#%%%%%%%%%%%%%%%%%%%

def process_file(file_path):
    
    # load the .mat file, squeezing all the useless Matlab extra dimensions
    s = scipy.io.loadmat(file_path, squeeze_me=True)
    
    minimum_frequency = s['starting_fft_frequency'] # 0 Hz
    maximum_frequency = 128 # TODO hardcoded # TODO magari si può ottenere dal tempo di sottocampionamento # TODO 1/(2*s['subsampling_time'])
    frequency_interval = maximum_frequency - minimum_frequency
    fft_lenght = s['fft_lenght'] # t_FFT = 8192 seconds
    frequency_resolution = s['frequency_resolution'] # 1/t_FFT = 1/8192
    reduction_factor = s['reduction_factor'] # TODO capire come è definito
    
    frequencies = numpy.linspace(start=minimum_frequency, stop=maximum_frequency, num=int(frequency_interval/frequency_resolution))
    # do NOT use numpy.arange(start=minimum_frequency, stop=maximum_frequency, step=frequency_resolution) because of half-open interval!
    subsampled_frequencies = numpy.linspace(start=minimum_frequency, stop=maximum_frequency, num=int(frequency_interval/(frequency_resolution*reduction_factor))) # TODO farlo in maniera più elegante
    
    number_of_zeros = s['number_of_zeros'] # TODO controllarne correttezza relativamente a percentage_of_zeros
    unilateral_number_of_samples = s['unilateral_number_of_samples']
    percentage_of_zeros = s['percentage_of_zeros'] # TODO attenzione che percentage_of_zeros NON è uguale a number_of_zeros/unilateral_number_of_samples (anche perché il rapporto viene a volte > 1 (cosa assurda))
    
    # TODO 1/sqrt
    total_normalization = numpy.sqrt(2)*s['normalization_factor']*s['window_normalization']*numpy.sqrt(1 - percentage_of_zeros)
    
    scaling_factor = s['scaling_factor'] # arbitrary factor used sometimes to rescale the data
    
    # TODO ex memory error
    # TODO valori diversi se si mette all'esterno il quadrato della normalizzazione
    power_spectrum = numpy.square(numpy.abs(s['fft_data']*total_normalization))#*scaling_factor
    # TODO fft unilatera?
    
    # autoregressive_spectrum and periodogram are stored in the file as square roots, so we need to make the square of them
    autoregressive_spectrum = numpy.square(s['autoregressive_spectrum'])#*scaling_factor
    periodogram = numpy.square(s['periodogram'])#*scaling_factor
    
    # frequency band selection
    # the best/cleaner frequency interval is roughly the region between 80 Hz and 120 Hz
    # so, that will be the only band we'll analyze
    cleanest_frequency_band = numpy.logical_and(80 <= frequencies, frequencies <= 120)
    selected_power_spectrum = power_spectrum[cleanest_frequency_band]
    selected_frequencies = frequencies[cleanest_frequency_band]
    
    cleanest_subsampled_frequency_band = numpy.logical_and(80 <= subsampled_frequencies, subsampled_frequencies <= 120)
    selected_autoregressive_spectrum = autoregressive_spectrum[cleanest_subsampled_frequency_band]
    selected_periodogram = periodogram[cleanest_subsampled_frequency_band]
    selected_subsampled_frequencies = subsampled_frequencies[cleanest_subsampled_frequency_band]
    
    # untill now, we have filtered and selected frequencies. so it was useful to have the main axis of the matrices on the dimension "frequency"
    # from here on, we will need to iterate over "time". so it's useful to transpose everything
    power_spectrum = numpy.transpose(power_spectrum)
    autoregressive_spectrum = numpy.transpose(autoregressive_spectrum)
    periodogram = numpy.transpose(periodogram)
    selected_power_spectrum = numpy.transpose(selected_power_spectrum)
    selected_autoregressive_spectrum = numpy.transpose(selected_autoregressive_spectrum)
    selected_periodogram = numpy.transpose(selected_periodogram)
    
    # all the following filter selections are evaluated in the interesting frequency band only
    # in this way we do not waste good samples that have minor problems outside the region of interst
    
    # all-empty FFTs are immediately discarded 
    is_empty = numpy.all(selected_power_spectrum == 0, axis=1) # zeros in the frequency (Fourier) space
    
    
    
    
    
    goods = [16, 17, 18, 19, 20, 21, 63, 64, 75, 76, 77, 82, 83, 94]
    
    # TODO condizione sul percentage_of_zeros dei dati nel dominio del tempo
    # TODO percentage_of_zeros[goods]
    
    
    
    
    
    # given the fact that out current data are really dirty, we place a condition on the median of the autoregressive spectrum, to be sure that it lies in the correct range # TODO levare questo vincolo quando i dati saranno migliori # TODO poi dividere per sps col 128
    # the periodogram can be higher than the autoregressive spectrum, because it suffers when there are bumps and unwanted impulses in the time domain
    # the median is more robust than the average
    autoregressive_spectrum_median = numpy.median(selected_autoregressive_spectrum, axis=1)
    absolute_tolerance = 1e-7 # TODO fine tuned (seguendo i risultati della valutazione fatta ad occhio) (sarebbe meglio mettere differenza relativa, per essere maggiormente future-proof)
    is_in_the_usual_range = numpy.isclose(autoregressive_spectrum_median, 6.5e-7, atol=absolute_tolerance) # (6.5 ± 1) 10^-7
    is_out_of_usual_range = numpy.logical_not(is_in_the_usual_range)
    is_empty_or_unusual = numpy.logical_or(is_empty, is_out_of_usual_range)
    
    # autoregressive_spectrum and periodogram must be more or less the same in this flat area
    # they are different in the peaks, because by construction the autoregrerrive mean ignores them
    # the autoregressive_spectrum can follow the noise nonstationarities
    periodogram_median = numpy.median(selected_periodogram, axis=1)
    median_difference = autoregressive_spectrum_median - periodogram_median
    has_discrepancies = numpy.abs(median_difference) >= 1e-5 # max_difference = 10^-5 # TODO fine tuned (sarebbe meglio mettere differenza relativa, per essere maggiormente future-proof)
    
    is_flagged = numpy.logical_or(is_empty_or_unusual, has_discrepancies) # is_empty | is_out_of_usual_range | has_discrepancies
    is_science_ready = numpy.logical_not(is_flagged)
    # TODO farlo con numpy.any
    
    clean_power_spectrum = power_spectrum[is_science_ready]
    clean_autoregressive_spectrum = autoregressive_spectrum[is_science_ready]
    clean_periodogram = periodogram[is_science_ready]
    
    clean_selected_power_spectrum = selected_power_spectrum[is_science_ready]
    clean_selected_autoregressive_spectrum = selected_autoregressive_spectrum[is_science_ready]
    clean_selected_periodogram = selected_periodogram[is_science_ready]
    
    gps_time = astropy.time.Time(val=s['gps_time'], format='gps', scale='utc')
    gps_time_values = gps_time.value
    # ISO 8601 compliant date-time format: YYYY-MM-DD HH:MM:SS.sss
    iso_time_values = gps_time.iso
    # time of the first FFT of this file
    human_readable_start_time = iso_time_values[0]
    
    clean_iso_time_values = iso_time_values[is_science_ready]
    
    detector = s['detector']

    fft_index = s['fft_index']
    print('Processing', file_path)
    print('Good spectra:', len(fft_index[is_science_ready]),'out of',len(fft_index))
    
    # TODO controllare che il valore medio sul plateau (10^-6) sia consistente con quanto scritto nella mia tesina
    
    plot_it = False
    
    if plot_it:
        for spectrum in clean_selected_power_spectrum:
            pyplot.figure()
            pyplot.grid()
            pyplot.semilogy(selected_frequencies, spectrum)
    
    if plot_it:
        for i in range(len(fft_index[is_science_ready])): # TODO iterare direttamente su fft_index usando xarray
            fig, [total, zoom] = pyplot.subplots(nrows=2, ncols=1, figsize=[10,10])
            #fig.suptitle(...)
            total.grid()
            zoom.grid()
            total.semilogy(frequencies, clean_power_spectrum[i], 
                           label='Normalized Power FFT')
            total.semilogy(subsampled_frequencies, clean_autoregressive_spectrum[i], 
                           color='#cc0000', label='autoregressive spectrum')
            total.semilogy(subsampled_frequencies, clean_periodogram[i], 
                           color='black', label='periodogram')
            
            # draw a rectangle to highlight the zoomed part # TODO zorder
            total.add_patch(matplotlib.patches.Rectangle(xy=[80, 1e-12 ], width=120-80, height=1e-2-1e-12, 
                                                         fill=False, alpha=1.0, linewidth=3, edgecolor="darkgrey"))
            
            zoom.semilogy(selected_frequencies, clean_selected_power_spectrum[i], 
                          label='Normalized Power FFT')
            zoom.semilogy(selected_subsampled_frequencies, clean_selected_autoregressive_spectrum[i], 
                          color='#cc0000', label='autoregressive spectrum')
            zoom.semilogy(selected_subsampled_frequencies, clean_selected_periodogram[i], 
                          color='black', label='periodogram')
            total.set_xlabel('Frequency [Hz]')
            zoom.set_xlabel('Frequency [Hz]')
            # TODO total.set_xlabel(...) # TODO amplitude spectral density VS strain VS 1/sqrt(Hz) VS 1/Hz
            # TODO zoom.set_xlabel(...)
            total.set_title('{} O2 C00 {} (0 Hz - 128 Hz)'.format(detector, clean_iso_time_values[i]), size=16) # hardcoded
            #zoom.set_title('Zoomed spectrum: (80 Hz - 120 Hz)') # TODO
            # TODO mettere limiti in x da 0 a 128 e farli combaciare col bordo figura
            total.legend(loc='upper right')
        
        
    # set all the flagged values to zero
    power_spectrum[is_flagged] = 0
    selected_power_spectrum[is_flagged] = 0
    autoregressive_spectrum[is_flagged] = 0
    selected_autoregressive_spectrum[is_flagged] = 0
    periodogram[is_flagged] = 0
    selected_periodogram[is_flagged] = 0
    
    # create a unitary strusture to return
    
    selected_frequencies = numpy.single(selected_frequencies) # float32
    gps_time_values = numpy.single(gps_time_values) # float32
    
    coordinate_names = ['frequency','GPS_time','detector'] # 3 detectors, so we are preparing an RGB image
    coordinate_values = [selected_frequencies, gps_time_values, [detector]]
    attributes = {'FFT_lenght': fft_lenght,
                  'observing run': 'O2', # TODO hardcoded (estrarlo dal file path)
                  'calibration': 'C00', # TODO hardcoded
                  'maximum frequency': maximum_frequency, # TODO hardcoded
                  'start_ISO_time':human_readable_start_time} # TODO metterlo come attibuto del singolo spettrogramma
    
    spectrogram = xarray.DataArray(data=numpy.expand_dims(numpy.transpose(selected_power_spectrum), axis=-1), 
                                   dims=coordinate_names, 
                                   coords=coordinate_values) #, attrs=attributes) #name='immagine'
    locally_science_ready = xarray.DataArray(data=numpy.expand_dims(is_science_ready, axis=-1), 
                            dims=['GPS_time','detector'], 
                            coords=[gps_time_values, [detector]])
    
    dataset = xarray.Dataset(data_vars={'spectrogram':spectrogram, 'locally_science_ready':locally_science_ready}, 
                        coords={'frequency':selected_frequencies,'GPS_time':gps_time_values}, 
                        attrs=attributes)
    
    return dataset

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# TODO mettere cartella di destinazione quando si elaborano e convertono tutti i file

# TODO leggere parallelamente e concatenare secondo l'ordine dato dalla data della prima FFT

def process_folder(path):
    # TODO vedere se è un file o una cartella
#    if is_file:
#        file_path = path
#    if is_folder: # TODO else if
#        folder_path = path
    folder_path = path
    
    mat_files = glob.glob(folder_path + '*.mat') # non funziona con la data_dir generale
    mat_files = sorted(mat_files) # TODO per cercare di garantire la continuità dei valori di GPS_time
    
    datasets = []
    for mat_file in mat_files:
        datasets.append(process_file(mat_file))
    
    complete_dataset = xarray.concat(objs=datasets, dim='GPS_time')

    return complete_dataset


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

LIGO_Hanford_data_dir = '/storage/pss/ligo_h/sfdb/O2/128/' # TODO hardcoded
LIGO_Livingston_data_dir = '/storage/pss/ligo_l/sfdb/O2/128/' # TODO hardcoded


# TODO vedere se gli spettri selezionati sono aumentati ottimizzando i tagli ad occhio

# TODO poi leggere tutto /storage e dare groupby sui detector

LIGO_Hanford_complete_dataset = process_folder(LIGO_Hanford_data_dir)
LIGO_Livingston_complete_dataset = process_folder(LIGO_Livingston_data_dir)
# TODO manca Virgo

# TODO per ora tenere i dataset separati e poi unirli tutti in RGB concatenando lungo l'asse 'detector' # TODO renderlo variabile e non più attributo
    
    # TODO salvare tutto nella struttura e aggiornare il file .mat
    # TODO controllare che l'intero codice sia parallelo (eccetto per la parte di lettura file)
    
    # TODO rinominare lo script in data_preprocessing.py
    
RGB_dataset = xarray.concat(objs=[LIGO_Hanford_complete_dataset, LIGO_Livingston_complete_dataset], dim='detector')
# TODO rendere la dimensione Categorical invece che Object

# TODO invece che flag, ha più senso importare science_ready, perché più immadiatamente comprensibile a chi legge

globaly_science_ready = RGB_dataset.locally_science_ready.all(dim='detector') # check if all detectors are in science mode
RGB_dataset.update({'globaly_science_ready': globaly_science_ready})
#numpy.any(copia_flag, axis=1)
# numpy.any True = or (ma invece come metodo degli oggetti è molto poco chiaro (BUG: ci ho messo parecchio per capirlo))
# numpy.all True = and

# requires netCDF4 or [h5py + h5netcdf] python packages installed
RGB_dataset.to_netcdf('~/Desktop/RGB_dataset.netCDF4', format='NETCDF4')

#######################

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
pyplot.imshow(image.T, cmap='gray', norm=matplotlib.colors.LogNorm(vmax=1e-22, vmin=1e-32), aspect='auto', origin="lower", interpolation="none")
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




