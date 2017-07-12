
import numpy
import xarray
#from matplotlib import pyplot

H_dataset = xarray.open_mfdataset('/storage/users/Muciaccia/netCDF4/O2/C01/128Hz/LIGO Hanford*.netCDF4')

# /storage/pss/virgo/sfdb/VSR4/v0/FULL/256Hz-hp20 e poi sono nelle directory sotto a pezzi di 10 gg

#H_dataset = H_dataset.isel(GPS_time = slice(0,128), frequency = slice(0,256))

# TODO per ora generare un dataset fittizio per Virgo. poi prenderlo dai nuovi dati di O2 o dai vecchi dati di VSR4
V_dataset = H_dataset.copy()
V_dataset.update({'detector':['Virgo']}) # TODO inplace
#RGB_median = 3e-07, 2e-07, 2.5e-07
# std=1e-6 ?

white_noise = False
if white_noise: # TODO
    # TODO hardcoded
    # valori copiati da quelli degli altri interferometri funzionanti
    normalization_factor = 4.31583721365314e-05
    window_normalization = 1.2063332796096802
    
    noise_amplitude = 0.5e-7
    # TODO è verosimile questa intensità del rumore? o c'è qualche problema con le normalizzazioni?
    
    number_of_times = len(V_dataset.GPS_time)
    number_of_frequencies = len(V_dataset.frequency)
    
    # TODO BUG 1j 2j 3j ... sono tutti numeri complessi, senza che `j` sia definito. come è posibile? questa sintassi confonde l'utente
    random_complex_numbers = numpy.single(numpy.random.randn(number_of_frequencies, number_of_times))+1j*numpy.single(numpy.random.randn(number_of_frequencies, number_of_times))
    
    fft_data = random_complex_numbers * noise_amplitude * 1/numpy.sqrt(2) * 1/normalization_factor
    # TODO capire perché normalization_factor e sqrt(2) stanno prima a numeratore e poi a denominatore
    
    # TODO capire !!!
    total_normalization = 1 #numpy.sqrt(2)*normalization_factor*window_normalization
    
    power_spectrum = numpy.square(numpy.abs(fft_data*total_normalization))
    
    # TODO non ce la fa con la memmoria RAM!!!
    
    #power_spectrum = numpy.transpose(power_spectrum)
    
    # TODO la mediana deve fare circa 1e-6 o poco meno, come per gli altri due detector
    
    #pyplot.hist(numpy.log(power_spectrum).flatten(), bins=100)
    #pyplot.show()
    # TODO gli altri detector hanno min=-20 e max=-10

    power_spectrum = numpy.expand_dims(power_spectrum, axis=-1)

# tutti uni
power_spectrum = 1e-6 * numpy.ones_like(V_dataset.spectrogram)

V_dataset.update({'spectrogram':(['frequency','GPS_time','detector'],power_spectrum)})
#del V_spectrogram
V_locally_science_ready = numpy.ones_like(V_dataset.locally_science_ready).astype(bool) # all True
V_dataset.update({'locally_science_ready':(['GPS_time','detector'],V_locally_science_ready)})

V_dataset.to_netcdf('/storage/users/Muciaccia/fake_Virgo_dataset.netCDF4', format='NETCDF4')

#V_dataset.isel(GPS_time = slice(0,128), frequency = slice(0,256)).spectrogram.squeeze().plot()
#pyplot.show()


