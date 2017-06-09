
# read sfdb files
# the SFDB09 file format (Short FFT DataBase, 2009 specification) is developed by Sergio Frasca and Ornella Piccinni

import os
import numpy
import scipy.io
import matplotlib
from matplotlib import pyplot

# to read the SFDB09 data, we need a function (written by Pia Astone) defined inside the Snag Matlab package (written by Sergio Frasca)
# Snag is a Matlab data analysis toolbox oriented to gravitational-wave antenna data
# Snag webpage: http://grwavsf.roma1.infn.it/snag/
# version 2, released 12 May 2017
# installation instructions:
# http://grwavsf.roma1.infn.it/snag/Snag2_UG.pdf



### %matplotlib inline



print('detectors: \t LIGO Hanford, LIGO Livingston')
print('observing run: \t O2') # most recent data
print('calibration: \t C00') # first calibration, then we will use C01
print('SFDB band: \t "256" (from 10 Hz to 128 Hz)') # TODO controllare
# possible SFDB bands: 256, 512, 1024, 2048

data_dir = '/storage'

LIGO_Hanford_data_dir = '/storage/pss/ligo_h/sfdb/O2/128/'
LIGO_Livingston_data_dir = '/storage/pss/ligo_l/sfdb/O2/128/'
# TODO i dati in /storage sono delle copie di /data

files = os.listdir(LIGO_Livingston_data_dir) # restituisce solo i nomi dei file/cartelle e non il loro percorso

import fnmatch

os.path # TODO

mat_files = fnmatch.filter(files, '*.mat')

import glob

LIGO_Livingston_mat_files = glob.glob(LIGO_Livingston_data_dir + '*.mat') # non funziona con la data_dir generale

# TODO ordinare la lista di file in ordine cronologico
mat_files = sorted(mat_files)

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

################################



# TODO vedere modulo interno di python per l'elaborazione dei path

# file craeate_peakmap.m

# fare rumore vero con e senza soglia

#altro_path = './dati_di_prova/altri/L1:GDS-CALIB_STRAIN_20161228_110935.SFDB09'

altro_path = './storage/pss/ligo_l/sfdb/O2/128/L1:GDS-CALIB_STRAIN_20161209_120255.SFDB09.mat'

# TODO scipy does not support v7.3 mat-files
# matlab --v7.3 files are hdf5 datasets
# import h5py 
# f = h5py.File('somefile.mat','r') 
# data = f.get('data/variable1') 
# data = np.array(data)

# periodogram = tps
# autoregressive_spectrum = sps
# data = sft

show_time_data = True



def process_file(file_path):
    
    # load the .mat file, squeezing all the useless Matlab extra dimensions
    a = scipy.io.loadmat(file_path, squeeze_me=True)
    
    scaling_factor = a['scaling_factor']
    autoregressive_spectrum = numpy.square(a['autoregressive_spectrum'])*scaling_factor # TODO
    periodogram = numpy.square(a['periodogram'])*scaling_factor # TODO
    fft_data = a['data'] # TODO complex64 # TODO valutare se rinominarlo nel file .mat # TODO fft unilatera?
    normalization_factor = a['normalization_factor']
    window_normalization = a['window_normalization']
    number_of_zeros = a['number_of_zeros'] # TODO controllare (anche con percentage_of_zeros)
    unilateral_number_of_samples = a['unilateral_number_of_samples']
    #frequency_resolution = a['frequency_resolution'] # 1/8192
    reduction_factor = a['reduction_factor'] # TODO capire come è definito
    fft_index = a['fft_index']
    detector = a['detector']
    
    # del a # poi automaticamente eliminato dal garbage collector della funzione
    
    minimum_frequency = 0 # TODO 10 Hz?
    maximum_frequency = 128
    frequency_interval = maximum_frequency - minimum_frequency
    t_FFT = 8192 # seconds
    frequency_resolution = 1/t_FFT
    # len(fft_data) = frequency_interval/frequency_resolution
    # TODO reduction_factor = (da 0 a 128 Hz)/frequency_resolution ?
    
    frequencies = numpy.linspace(start=minimum_frequency, stop=maximum_frequency, num=int(frequency_interval/frequency_resolution))
    #numpy.arange(start=minimum_frequency, stop=maximum_frequency, step=frequency_resolution) # do not use it! half-open interval
    reduced_frequencies = numpy.linspace(start=minimum_frequency, stop=maximum_frequency, num=int(frequency_interval/(frequency_resolution*reduction_factor)))
    
    total_normalization = numpy.sqrt(2)*normalization_factor*window_normalization*numpy.sqrt(1 - number_of_zeros/unilateral_number_of_samples)
    # TODO number_of_zeros/unilateral_number_of_samples = a['percentage_of_zeros'] ? 
    
    # TODO memory error
    power_spectrum = numpy.square(numpy.abs(fft_data*total_normalization))*scaling_factor # TODO vs norm^2
    
    # frequency band selection
    # the best/cleaner frequency interval is roughly the region between 80 Hz and 120 Hz
    cleanest_frequency_band = numpy.logical_and(80 <= frequencies, frequencies <= 120)
    selected_power_spectrum = power_spectrum[cleanest_frequency_band]
    selected_frequencies = frequencies[cleanest_frequency_band]
    
    cleanest_reduced_frequency_band = numpy.logical_and(80 <= reduced_frequencies, reduced_frequencies <= 120)
    selected_autoregressive_spectrum = autoregressive_spectrum[cleanest_reduced_frequency_band]
    selected_periodogram = periodogram[cleanest_reduced_frequency_band]
    selected_reduced_frequencies = reduced_frequencies[cleanest_reduced_frequency_band]
    
    # fino a qui serve che l'indice principale sia sulle frequenze e non sull'indice di fft, poiché è utile filtrare o selezionare le frequenze
    
    # da qui in poi invece ci servirà iterare sull'indice di fft, quindi per comodità trasponiami tutti gli array
    
    power_spectrum = numpy.transpose(power_spectrum)
    selected_power_spectrum = numpy.transpose(selected_power_spectrum)
    autoregressive_spectrum = numpy.transpose(autoregressive_spectrum)
    selected_autoregressive_spectrum = numpy.transpose(selected_autoregressive_spectrum)
    periodogram = numpy.transpose(periodogram)
    selected_periodogram = numpy.transpose(selected_periodogram)
    
    # tutti i filtri sono applicati nella banda di interesse, non fuori (per non buttare inutilmente campioni)
    
    is_empty = numpy.all(selected_power_spectrum == 0, axis=1)
    is_out_of_usual_range = selected_power_spectrum.max(axis=1) > 1e-22 # TODO fine tunined
    
    # autoregressive_spectrum and periodogram must be more or less the same
    # periodogramma e spettro autoregressivo sono diversi sui picchi (perché la media autoregressiva li ignora per costruzione)
    absolute_tolerance = 1e-25 # TODO fine tuned
    has_discrepancies = numpy.any(numpy.logical_not(numpy.isclose(selected_autoregressive_spectrum, selected_periodogram, atol=absolute_tolerance)), axis=1)
    
    is_flagged_old = numpy.logical_or(is_empty, is_out_of_usual_range)
    is_flagged = numpy.logical_or(is_flagged_old, has_discrepancies) # is_empty | is_out_of_usual_range | has_discrepancies
    
    science_ready = numpy.logical_not(is_flagged)
    
    clean_power_spectrum = power_spectrum[science_ready]
    clean_autoregressive_spectrum = autoregressive_spectrum[science_ready]
    clean_periodogram = periodogram[science_ready]
    
    clean_selected_power_spectrum = selected_power_spectrum[science_ready]
    clean_selected_autoregressive_spectrum = selected_autoregressive_spectrum[science_ready]
    clean_selected_periodogram = selected_periodogram[science_ready]
    
    plot_it = False
    if plot_it:
        for spectrum in clean_selected_power_spectrum:
            pyplot.figure()
            pyplot.grid()
            pyplot.semilogy(selected_frequencies, spectrum)
    
    # TODO vedere come creare direttamente array booleani in numpy (esempio: numpy.false(shape=(10,20)))
    # is_flagged = a['is_flagged'].astype(bool)
    
    # TODO iterare pure sulla dimensione dell'indice (usando xarray?)
    
    # TODO autoregressive_mean vs autoregressive_spectrum
    
    # risoluzione 8192/2 secondi per non poter mai uscire dal canale in frequenza
    
    # TODO dilatare (raddoppiare) la scala temporale
    
    if plot_it:
        for i in range(len(fft_index[science_ready])): # TODO iterare direttamente su fft_index usando xarray
            fig, [total, zoom] = pyplot.subplots(nrows=2, ncols=1, figsize=[10,10])
            fig.suptitle('{} UTC: data QUI'.format(detector), size=16) # TODO
            total.grid()
            zoom.grid()
            total.semilogy(frequencies, clean_power_spectrum[i], 
                           label='Normalized Power FFT')
            total.semilogy(reduced_frequencies, clean_autoregressive_spectrum[i], 
                           color='magenta', label='autoregressive mean') # TODO aggiustare colore
            total.semilogy(reduced_frequencies, clean_periodogram[i], 
                           color='black', label='periodogram') # TODO aggiustare colore
            
            # draw a rectangle to highlight the zoomed part # TODO zorder
            total.add_patch(matplotlib.patches.Rectangle(xy=[80, 1e-32 ], width=120-80, height=1e-22-1e-32, 
                                                         fill=False, alpha=1.0, linewidth=3, edgecolor="darkgrey"))
            
            zoom.semilogy(selected_frequencies, clean_selected_power_spectrum[i], 
                          label='Normalized Power FFT')
            zoom.semilogy(selected_reduced_frequencies, clean_selected_autoregressive_spectrum[i], 
                          color='magenta', label='autoregressive mean') # TODO aggiustare colore
            zoom.semilogy(selected_reduced_frequencies, clean_selected_periodogram[i], 
                          color='black', label='periodogram') # TODO aggiustare colore
            total.set_xlabel('Frequency [Hz]')
            zoom.set_xlabel('Frequency [Hz]')
            total.set_title('title')
            zoom.set_title('title')
            total.legend(loc='upper right')
        
        #from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
        #
        #fig = plt.figure(1)
        #ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])
        #
        #r, g, b = get_rgb() # r,g,b are 2-d images
        #ax.imshow_rgb(r, g, b,
        #              origin="lower", interpolation="nearest")
        
        
        # TODO fine tuning dei parametri di selezione
        
        # False,  True,  True,  True,  True,  True, False, False, False,
        # True, False,  True,  True,  True,  True,  True,  True,  True, False
        
    # set all the flagged values to zero
    power_spectrum[is_flagged] = 0
    selected_power_spectrum[is_flagged] = 0
    autoregressive_spectrum[is_flagged] = 0
    selected_autoregressive_spectrum[is_flagged] = 0
    periodogram[is_flagged] = 0
    selected_periodogram[is_flagged] = 0
    
    # TODO salvare tutto nella struttura e aggiornare il file .mat
    # TODO controllare che l'intero codice sia parallelo (eccetto per la parte di lettura file
    
    print('Good spectra:', len(fft_index[science_ready]),'out of',len(fft_index))
    
    return selected_power_spectrum, selected_frequencies, fft_index, science_ready, detector #, time


#########################


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


# TODO Maria Alessandra Papa
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




