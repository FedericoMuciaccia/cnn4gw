
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
import xarray
import scipy.io
import scipy.stats
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

file_path = altro_path = '/storage/users/Muciaccia/mat/O2/C01/128Hz/L/01-Apr-2017 01:07:58.000000.mat'
H_path = '/storage/users/Muciaccia/mat/O2/C01/128Hz/H/01-Apr-2017 01:07:58.000000.mat'

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
    
    # scaling_factor = s['scaling_factor'] # arbitrary factor used sometimes to rescale the data
    
    # TODO ex memory error
    # TODO valori diversi se si mette all'esterno il quadrato della normalizzazione
    power_spectrum = numpy.square(numpy.abs(s['fft_data']*total_normalization))#*scaling_factor
    # don't worry if an overflow RuntimeWarning will be printed by numpy: see below
    # TODO fft unilatera?
    
    #total_normalization = total_normalization.astype(numpy.float64)
    #total_normalization = numpy.double(total_normalization)
    # float64 slows down computation and cannot be handled by GPU
    # so we are forced to take into account the possibility of overflow and truncation errors (RuntimeWarning: overflow)
    # replace the eventual infinities with the maximum float32 number
    power_spectrum[numpy.isinf(power_spectrum)] = numpy.finfo(numpy.float32).max # float32_max = 3.4028235e+38
    
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
    is_empty = numpy.all(selected_power_spectrum == 0, axis=1) # all zeros in the frequency (Fourier) domain
    is_not_empty = numpy.logical_not(is_empty)
    
    has_not_many_temporal_holes = percentage_of_zeros < 0.2 # less than 20% zeros in the time domain
    
    
    #goods = [16, 17, 18, 19, 20, 21, 63, 64, 75, 76, 77, 82, 83, 94]
    
    # TODO percentage_of_zeros[goods]
    # TODO la condizione sulla differenza percentuale tra spettro autoregressivo e periodogramma mi sembra quella più solida e generale per essere future-proof e per tener conto dei miglioramenti nel tempo del rumore.
    # TODO NON è però abbastanza: rimangono degli spettri spuri
    # TODO sui picchi però c'è sempre una differenza, per cui forse sarebbe indicato fare prima il whitening
    
    
    
    # given the fact that out current data are really dirty, we place a condition on the median of the autoregressive spectrum, to be sure that it lies in the correct range # TODO levare questo vincolo quando i dati saranno migliori # TODO poi dividere per sps col 128
    # the periodogram can be higher than the autoregressive spectrum, because it suffers when there are bumps and unwanted impulses in the time domain
    # the median is more robust than the average
    autoregressive_spectrum_median = numpy.median(selected_autoregressive_spectrum, axis=1)
    #absolute_tolerance = 1e-7 # TODO fine tuned (seguendo i risultati della valutazione fatta ad occhio) (sarebbe meglio mettere differenza relativa, per essere maggiormente future-proof)
    #is_in_the_usual_range = numpy.isclose(autoregressive_spectrum_median, 6.5e-7, atol=absolute_tolerance) # (6.5 ± 1) * 10^-7
    #is_out_of_usual_range = numpy.logical_not(is_in_the_usual_range)
    #is_empty_or_unusual = numpy.logical_or(is_empty, is_out_of_usual_range)
    # TODO farlo con numpy.any()
    
    # autoregressive_spectrum and periodogram must be more or less the same in this flat area
    # they are different in the peaks, because by construction the autoregrerrive mean ignores them
    # the autoregressive_spectrum can follow the noise nonstationarities
    periodogram_median = numpy.median(selected_periodogram, axis=1)
    #median_difference = autoregressive_spectrum_median - periodogram_median
    #has_discrepancies = numpy.abs(median_difference) >= 1e-5 # max_difference = 10^-5 # TODO fine tuned (sarebbe meglio mettere differenza relativa, per essere maggiormente future-proof)
    
    #is_flagged = numpy.logical_or(is_empty_or_unusual, has_discrepancies) # is_empty | is_out_of_usual_range | has_discrepancies
    #is_science_ready = numpy.logical_not(is_flagged)
    # TODO farlo con numpy.any
        
    # TODO il valore basale del rumore è diverso per tutti e 3 i detector
    is_consistent = numpy.isclose(periodogram_median, autoregressive_spectrum_median, rtol=0.1) # relative_tolerance = 10% # TODO fine tuned
    # TODO MA gli vanno levati gli is_empty (perché la tolleranza relativa con gli zeri ha problemi) (is_close and not is_empty)
    # TODO new elementwise comparison: numpy.equal
    
    # la mediana è più resistente della media rispetto alla presenza di forti outliers
    # median: 'middle' value
    # mode: most common value
    # extreme outliers change the values of mean and variance
    
    # l'autoregressive_spectrum segue meglio i dati rispetto al periodogramma
    # il secondo quartile è la mediana dai dati
    # interquartile_ratio/2 è l'equivalente della sigma (standard deviation) ma al 50% invece che al 68%
    goodness_constraints = numpy.all([is_not_empty, has_not_many_temporal_holes, is_consistent], axis=0) # check if all conditions are satisfied (like with logical and)
    
    # if there isn't any good FFT (that is: if all FFTs are bad)
    # (this if statement is required because in the calculation of the median we cannot divide by zero)
    if numpy.all(goodness_constraints == False): #if not numpy.any(goodness_constraints):
        is_science_ready = goodness_constraints # all False
    else:
        middle_value = numpy.median(autoregressive_spectrum_median[goodness_constraints])
        is_in_the_usual_range = numpy.isclose(autoregressive_spectrum_median, middle_value, rtol=0.5) # relative_tolerance = 50% # TODO fine tuned

        #numpy.all([is_consistent, is_not_empty], axis=0)
        is_science_ready = numpy.logical_and(goodness_constraints, is_in_the_usual_range)
    is_flagged = numpy.logical_not(is_science_ready)
    
    # TODO plottare gli istogrammi (magari 2D sui 2 detector) per trovare i tagli ottimali
    
    
    detector = s['detector']
    
    #clean_power_spectrum = power_spectrum[is_science_ready]
    #clean_autoregressive_spectrum = autoregressive_spectrum[is_science_ready]
    #clean_periodogram = periodogram[is_science_ready]
    
    #clean_selected_power_spectrum = selected_power_spectrum[is_science_ready]
    #clean_selected_autoregressive_spectrum = selected_autoregressive_spectrum[is_science_ready]
    #clean_selected_periodogram = selected_periodogram[is_science_ready]
    
    
    
    
#    if detector == 'Virgo': # TODO temporary hack for the VSR4 dataset
#        desired_gps_start_time = astropy.time.Time(val='2017-01-01 00:00:00.000', format='iso', scale='utc').gps
#        actual_gps_start_time = astropy.time.Time(val='2011-06-03 10:26:59.000', format='iso', scale='utc').gps
#        gps_time_shift = desired_gps_start_time - actual_gps_start_time
#        s['gps_time'] = s['gps_time'] + gps_time_shift # TODO c'è qui una differenza finale di 18 secondi
    
    # TODO il float32 è insufficiente a rappresentare il tempo GPS con la precisione dovuta, perché perde le ultime due cifre del tempo GPS (decine ed unità)
    # TODO dato che servirà calcolare su GPU e che dunque serve il float32, propondo di ridefinire lo standard temporale GPS a partire dall' 1 gennaio 2000, invece che dal 6 gennaio 1980, chiamandolo millennium_time
    # TODO vedere le time series di pandas e xarray come risolvono il problema
    # TODO data/values indexes labels/coordinates axis/dimensions
    # TODO pandas.CategoricalIndex pandas.IndexSlice pandas.IntervalIndex pandas.MultiIndex pandas.SparseArray pandas.TimedeltaIndex
    
    
    gps_time = astropy.time.Time(val=s['gps_time'], format='gps', scale='utc')
    gps_time_values = gps_time.value.astype(numpy.float32)
    
    # ISO 8601 compliant date-time format: YYYY-MM-DD HH:MM:SS.sss
    iso_time_values = gps_time.iso
    # time of the first FFT of this file
    human_readable_start_time = iso_time_values[0]
    
    #clean_iso_time_values = iso_time_values[is_science_ready]

    fft_index = s['fft_index'] - 1 # index in python start from 0 instead on 1, as in Matlab
    print('Processing', file_path)
    print('Good spectra:', len(fft_index[is_science_ready]),'out of',len(fft_index))
    
    # TODO controllare che il valore medio sul plateau (10^-6) sia consistente con quanto scritto nella mia tesina    
    
    
#    hist_H = []
#    hist_L = []
#    hist_V = []
#    if detector == 'LIGO Hanford':
#        hist_H.append(numpy.log(autoregressive_spectrum_median))
#    if detector == 'LIGO Livingston':
#        hist_L.append(numpy.log(autoregressive_spectrum_median))
#    pyplot.hist2d(x=H, y=L, bins=100, cmap='viridis')
#    #pyplot.hist(numpy.log(numpy.median(clean_selected_periodogram, axis=1)), bins=100)
#    #pyplot.show()
#    # TODO fare istogrammi 2D per dimostrare la bontà delle superfici di separazione e dei tagli fatti
#    # TODO mettere legenda per l'immagine a colori, col cerchio della sintesi additiva in SVG (su un'immagine quadrata di sfondo nero) con le lettere indicative dei detector (fare delle prove con un'immagine creata ad hoc). controllare che dove mancano tutti i dati le righe verticali siano nere (dovute agli zeri) e non bianche (dovute ai NaN)
#    # TODO mettere finestra in frequenza per le immagini solo da 80 a 120 Hz e non su tutta la banda di 128 Hz (ed eventualmente ricomputare l'intervallo di 0.1 Hz)
#    # TODO portare la creazione delle immagini su xarray, in modo che il calcolo possa essere fatto senza vincoli di memoria su qualsiasi computer
#    # TODO vedere computazione out-of-memory per big-data su tensorflow
#    # TODO posticipare la normalizzazione logaritmica a dopo che si fanno le injections
#    # TODO classificare le immagini degli spettri per mostrare tutti i vari casi possibili (compreso quello degli zeri temporali, antitrasformando in Fourier)
#    # TODO fare plot di k-fold validation con numpy.percentile([5, 25, 50, 75 95], data) in modo da evere la linea di mediana e le linee con confidenza al 50% e al 90%, come fanno gli astrofisici (sensatamente, ora capisco). [5, 25, 50, 75 95] = [50-90/2, 50-50/2, 50, 50+50/2, 50+90/2]
#    # TODO poi, dopo la classificazione (trigger), fare regressione per paramenter estimation
#    # TODO Ricci vuole dei tool per studiare il rumore online (nella fase di commissioning dell'interferometro)
#    # TODO chiedere a Ornella di generare i dati più recenti
#    # TODO mettere i dati di Virgo di VSR4 (o gli ultimi di O2)
#    # TODO valutare se creare i file .netCDF4 direttamente in Matlab, in modo da risparmiare lo spazio dei file .mat
#    img = numpy.zeros([100, 100, 3])
#    img[19:79,0:59,0] = 1
#    img[39:99,19:79,1] = 1
#    img[0:59,39:99,2] = 1
#    # img[0:59,0:59,0] = 1
#    # img[19:79,19:79,1] = 1
#    # img[39:99,39:99,2] = 1
#    pyplot.imshow(img, origin="lower", interpolation="none")
#    pyplot.show()
    
    
    plot_it = False
    
    if plot_it:
        for spectrum in selected_power_spectrum[is_science_ready]: # clean_selected_power_spectrum
            pyplot.figure()
            pyplot.grid()
            pyplot.semilogy(selected_frequencies, spectrum)
            #pyplot.savefig('{}.svg'.format(i))
            pyplot.show()
            pyplot.close()
    
    plot_it = False
    
    if plot_it:
        #for i in range(len(fft_index[is_science_ready])): # TODO iterare direttamente su fft_index usando xarray
        #@numpy.vectorize # TODO BUG: ripete il primo elemento
        def my_plot_figure(i):
            print(i)
            fig, [total, zoom] = pyplot.subplots(nrows=2, ncols=1, figsize=[10,10])
            #fig.suptitle(...)
            total.grid()
            zoom.grid()
            total.semilogy(frequencies, power_spectrum[i], 
                           label='Normalized Power FFT')
            total.semilogy(subsampled_frequencies, autoregressive_spectrum[i], 
                           color='#cc0000', label='autoregressive spectrum')
            total.semilogy(subsampled_frequencies, periodogram[i], 
                           color='black', label='periodogram')
            
            # draw a rectangle to highlight the zoomed part # TODO zorder
            total.add_patch(matplotlib.patches.Rectangle(xy=[80, 1e-12 ], width=120-80, height=1e-2-1e-12, 
                                                         fill=False, alpha=1.0, linewidth=3, edgecolor="darkgrey"))
            
            zoom.semilogy(selected_frequencies, selected_power_spectrum[i], 
                          label='Normalized Power FFT')
            zoom.semilogy(selected_subsampled_frequencies, selected_autoregressive_spectrum[i], 
                          color='#cc0000', label='autoregressive spectrum')
            zoom.semilogy(selected_subsampled_frequencies, selected_periodogram[i], 
                          color='black', label='periodogram')
            total.set_xlabel('Frequency [Hz]')
            zoom.set_xlabel('Frequency [Hz]')
            # TODO total.set_xlabel(...) # TODO amplitude spectral density VS strain VS 1/sqrt(Hz) VS 1/Hz
            # TODO zoom.set_xlabel(...)
            total.set_title('{} O2 C01 {} (0 Hz - 128 Hz)'.format(detector, iso_time_values[i]), size=16) # TODO hardcoded
            #zoom.set_title('Zoomed spectrum: (80 Hz - 120 Hz)') # TODO
            # TODO mettere limiti in x da 0 a 128 e farli combaciare col bordo figura
            total.legend(loc='upper right')
            #pyplot.show()
            pyplot.savefig('/storage/users/Muciaccia/media/spectra selection/{}.jpg'.format(i))
            pyplot.close()
        
        my_plot_figure = numpy.frompyfunc(my_plot_figure, 1,1) # TODO hack per vettorializzare
        #print(fft_index[is_science_ready])
        my_plot_figure(fft_index[is_science_ready])
        #my_plot_figure(63)
        # good_discarded = 97? // 48 95
        # bad_selected = 66? //
        # dunque questi criteri di selezione possono portare un 1% o più di falsi positivi e falsi negativi
        # TODO ottimizzare i tagli (oppure farli fare direttamente alla macchina)
        # TODO plottare falsi positivi e falsi negativi
        # TODO plottare anche i dati con i buchi nel tempo, per far vedere anche quel criterio di selezione
        
    
    # TODO BUG di numpy: si ripete il primo indice
    #@numpy.vectorize
    #def plot_figure(fft_index): # TODO così dovrebbe essere vettoriale
    #    print(fft_index)
    #plot_array = numpy.frompyfunc(plot_figure, 1,1)
    
    # TODO BUG di python 
    # la definizione di funzioni dovrebbe essere effettuata tramite istanziazioni della classe function (tipo javascript?), che deve poter essere sostituita con altri tipi di classi più raffinate come le vectorized_function (numpy ufunc)
    
    
    
    
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
                  'observing_run': 'O2', # TODO hardcoded (estrarlo dal file path)
                  'calibration': 'C01', # TODO hardcoded
                  'maximum_frequency': maximum_frequency, # TODO hardcoded
                  'start_ISO_time':human_readable_start_time} # TODO metterlo come attibuto del singolo spettrogramma (e levarlo dal file complessivo)
    # TODO mettere anche tutti gli altri attributi interessanti come are_fft_interlaced = True
    
    spectrogram = xarray.DataArray(data=numpy.expand_dims(numpy.transpose(selected_power_spectrum), axis=-1), 
                                   dims=coordinate_names, 
                                   coords=coordinate_values) #, attrs=attributes) #name='immagine'
    locally_science_ready = xarray.DataArray(data=numpy.expand_dims(is_science_ready, axis=-1), 
                            dims=['GPS_time','detector'], 
                            coords=[gps_time_values, [detector]]) # TODO [detector] VS detector
    
    dataset = xarray.Dataset(data_vars={'spectrogram':spectrogram, 'locally_science_ready':locally_science_ready}, 
                        coords={'frequency':selected_frequencies,'GPS_time':gps_time_values}, 
                        attrs=attributes)
    
    return dataset
    # TODO make an option to return the raw dataset (converted in netCDF4 format) without the flagged values putted to zero


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#H_path = '/storage/users/Muciaccia/mat/O2/C01/128Hz/H/24-Dec-2016 08:10:23.000000.mat'

#/home/federico/.local/lib/python3.5/site-packages/numpy/core/fromnumeric.py:2909: RuntimeWarning: Mean of empty slice.
#  out=out, **kwargs)
#/home/federico/.local/lib/python3.5/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in true_divide
#  ret = ret.dtype.type(ret / rcount)

#file_path = L_path = '/storage/users/Muciaccia/mat/O2/C01/128Hz/L/05-Mar-2017 18:48:14.000000.mat'

#convert_mat_to_netCDF4.py:140: RuntimeWarning: overflow encountered in square
#  power_spectrum = numpy.square(numpy.abs(s['fft_data']*total_normalization))#*scaling_factor



#a = process_file(altro_path)

#exit()

# TODO rendere lo script una libreria, in modo da poter importare le funzioni ad esempio per fare singoli plot

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
    
    # TODO funziona!!!
    mat_files = glob.glob(folder_path + '**/*.mat', recursive=True) # TODO segue correttamente anche i link simbolici (dunque attenzione ai loop)
    mat_files = sorted(mat_files) # TODO per cercare di garantire la continuità dei valori di GPS_time
    
#    datasets = []
    for mat_file in mat_files: # TODO questo ciclo è totalmente parallelizzabile
#        datasets.append(process_file(mat_file))
        dataset = process_file(mat_file)
        detector = str(numpy.squeeze(dataset.detector.values))
        print('Saving /storage/users/Muciaccia/netCDF4/O2/C01/128Hz/{} {}.netCDF4'.format(detector, dataset.start_ISO_time)) # TODO hardcoded
        
        dataset.to_netcdf('/storage/users/Muciaccia/netCDF4/O2/C01/128Hz/{} {}.netCDF4'.format(detector, dataset.start_ISO_time), format='NETCDF4') # TODO hardcoded # TODO non crea da solo le sottocartelle
#    complete_dataset = xarray.concat(objs=datasets, dim='GPS_time')
#    return complete_dataset


# TODO ricontrollare criteri di selezione con la nuova calibrazione
# TODO anche perché i vari detector possono avere valori basali differenti, che quindi fanno scartare la mediana (penso soprattutto per Hanford)

#process_folder('/storage/users/Muciaccia/mat/')
process_folder('/storage/users/Muciaccia/mat/O2/C01/128Hz/V/')

exit()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def mat_to_netCDF4(detector_paths):
    pass

# LIGO Hanford
H_data_dir = '/storage/users/Muciaccia/mat/O2/C01/128Hz/H/' # TODO hardcoded

# LIGO Livingston
L_data_dir = '/storage/users/Muciaccia/mat/O2/C01/128Hz/L/' # TODO hardcoded

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
        dataset.update({'globally_science_ready': globally_science_ready}) # TODO inplace=True
        
        print('Saving /storage/users/Muciaccia/netCDF4/O2/C00/128Hz/{}.netCDF4'.format(dataset.start_ISO_time)) # TODO hardcoded
        
        dataset.to_netcdf('/storage/users/Muciaccia/netCDF4/O2/C00/128Hz/{}.netCDF4'.format(dataset.start_ISO_time), format='NETCDF4') # TODO hardcoded # TODO non crea da solo le sottocartelle

# TODO controllare i fare float64 per i tempi e per le frequenze

# TODO posticipare l'operazione di calcolo di globallu_science_ready perché per ragioni di spazio non riesco a salvare entrambi i dataset contemporaneamente sul disco
# TODO oppure chiamare da python3 il Matlab engine e convertire in .netCDF4 ogni file singolarmente, evitando dunque di dover salvare i file .mat su disco
# TODO
# data_preprocessing.py
#     per ogni singolo file:
#         Matlab engine: SFDB09 -> mat   
#         import convert_to_netCDF4: mat -> netCDF4

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


