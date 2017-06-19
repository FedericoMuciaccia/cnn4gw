
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
    is_in_the_usual_range = numpy.isclose(autoregressive_spectrum_median, 6.5e-7, atol=absolute_tolerance) # (6.5 ± 1) * 10^-7
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
    
    # create a unitary structure to return
    
    selected_frequencies = numpy.single(selected_frequencies) # float32
    gps_time_values = numpy.single(gps_time_values) # float32
    
    coordinate_names = ['frequency','GPS_time','detector'] # 3 detectors, so we are preparing an RGB image
    coordinate_values = [selected_frequencies, gps_time_values, [detector]]
    attributes = {'FFT_lenght': fft_lenght,
                  'observing run': 'O2', # TODO hardcoded (estrarlo dal file path)
                  'calibration': 'C00', # TODO hardcoded
                  'maximum frequency': maximum_frequency, # TODO hardcoded
                  'start_ISO_time':human_readable_start_time} # TODO metterlo come attibuto del singolo spettrogramma (e levarlo dal file complessivo)
    # TODO mettere anche tutti gli altri attributi interessanti come are_fft_interlaced = True
    
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
    
    mat_files = glob.glob(folder_path + '*.mat') # TODO non funziona con la data_dir generale
    mat_files = sorted(mat_files) # TODO per cercare di garantire la continuità dei valori di GPS_time
    
    datasets = []
    for mat_file in mat_files:
        datasets.append(process_file(mat_file))
    
    complete_dataset = xarray.concat(objs=datasets, dim='GPS_time')

    return complete_dataset


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def mat_to_netCDF4(detector_paths):
    pass

# LIGO Hanford
H_data_dir = '/storage/users/Muciaccia/mat/O2/C00/128Hz/H/' # TODO hardcoded

# LIGO Livingston
L_data_dir = '/storage/users/Muciaccia/mat/O2/C00/128Hz/L/' # TODO hardcoded

# TODO iterare sui 3 detector
H_mat_files = sorted(glob.glob(H_data_dir + '*.mat'))
L_mat_files = sorted(glob.glob(L_data_dir + '*.mat'))

file_list = numpy.transpose(numpy.array([H_mat_files, L_mat_files]))

# TODO funzione_lettura(..., delete_original=False)

# TODO sperando che i files abbiano una corrispondenza 1 a 1 tra i 2 detector
# TODO ATTENZIONE! i file sono a coppie, ma non in ordine cronologico a causa del fatto che i file .mat hanno il nome che non comincia con l'anno!
# TODO valutare se magari salvare i file separatamente per detector, in modo da semplificare il codice e poi fare caricare e concatenare tutto automaticamente, in modo che non ci sia neanche bisgno di avere i file ordinati e dunque si possa anche totalmente parallelizzare il processo

# TODO farlo con numpy.apply_along_axis e funzione vettoriale di lettura
# TODO magari rendere parallela questa elaborazione
for file_H, file_L in file_list:
        ds_H = process_file(file_H)
        ds_L = process_file(file_L)
        dataset = xarray.concat(objs=[ds_H, ds_L], dim='detector')
        
        # check if all detectors are in science mode
        globally_science_ready = dataset.locally_science_ready.all(dim='detector')
        dataset.update({'globally_science_ready': globally_science_ready})
        
        print('Saving /storage/users/Muciaccia/netCDF4/O2/C00/128Hz/{}.netCDF4'.format(dataset.start_ISO_time)) # TODO hardcoded
        
        dataset.to_netcdf('/storage/users/Muciaccia/netCDF4/O2/C00/128Hz/{}.netCDF4'.format(dataset.start_ISO_time), format='NETCDF4') # TODO hardcoded # TODO non crea da solo le sottocartelle

# TODO controllare i fare float64 per i tempi e per le frequenze



exit()



# "256" è la banda di frequenza, perché va da -128 a +128 perché la trasformata di Fourier crea anche frequenze negative
# banda_completa = 1/subsampling_time
# banda_positiva = 1/2 * banda_completa # la banda positiva va da 0 a 128 Hz

# TODO vedere se gli spettri selezionati sono aumentati ottimizzando i tagli ad occhio

# BUG python e numpy: la numerazione a partire da 1 (invece che da 0) è molto più intuitiva e rilassante, perché rispecchia e riflette il modo comune di contare (ad esempio sulle dita delle mani)

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

globally_science_ready = RGB_dataset.locally_science_ready.all(dim='detector') # check if all detectors are in science mode
RGB_dataset.update({'globally_science_ready': globally_science_ready})
#numpy.any(copia_flag, axis=1)
# numpy.any True = or (ma invece come metodo degli oggetti è molto poco chiaro (BUG: ci ho messo parecchio per capirlo))
# numpy.all True = and

# requires netCDF4 or [h5py + h5netcdf] python packages installed
RGB_dataset.to_netcdf('~/Desktop/RGB_dataset.netCDF4', format='NETCDF4')


