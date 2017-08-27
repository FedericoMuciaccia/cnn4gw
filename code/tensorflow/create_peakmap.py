
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
from matplotlib import pyplot

#dataset = xarray.open_mfdataset('/storage/users/Muciaccia/netCDF4/O2/C01/128Hz/*.netCDF4')

dataset = xarray.open_dataset('prova.netCDF4')

# TODO mettere nel dataset i due attributi di 'descriprion' e 'reference' (con l'indirizzo web da cui reperire i dati)

# TODO vedere se la generazione di numeri random su tensorflow è parallela anche usando la CPU (diversamente da quanto avviene con numpy)

# xarray.tutorial.load_dataset('air_temperature')

# dataset.spectrogram.get_axis_num('GPS_time')

# my_dask_array.compute() # calcolo simbolico: viene calcolato il grafo

# TODO numpy.split(my_dask_array) VS dataset.groupby_bins VS pandas.MultiIndex

plotted_values = dataset.whitened_spectrogram.isel(GPS_time = 0).values
pyplot.hist(numpy.log(plotted_values), bins=100)
pyplot.vlines(x=numpy.log(numpy.median(plotted_values)), ymin=0, ymax=40000, label='median', color='red')
pyplot.vlines(x=numpy.log(numpy.percentile(plotted_values, [25.0, 75.0])), ymin=0, ymax=40000, label='50%', color='blue')
pyplot.vlines(x=numpy.log(numpy.percentile(plotted_values, [5.0, 95.0])), ymin=0, ymax=40000, label='90%', color='black')
pyplot.legend()
pyplot.show()

# log(std(...)) =! std(log(...))
# mediana e vari percentili cambiano invece pochissimissimo (proprio per come sono definiti)
# numpy.log(numpy.percentile(plotted_values, [5.0, 25.0, 50.0, 75.0, 95.0]))
# numpy.percentile(numpy.log(plotted_values), [5.0, 25.0, 50.0, 75.0, 95.0])


# TODO nello spettrogramma sbiancato ci sono comunque dei picchi dove il detector si acceca

#plot_RGB_image(numpy.squeeze(dataset.spectrogram)[0:256,0:128])
# TODO il rescaling interno del logaritmo è errato nel caso di dataset.whitened_spectrogram

pyplot.figure(figsize=[10,20])
numpy.log(dataset.spectrogram[0:256,0:128]).plot(vmin=-20, vmax=-5)
pyplot.show()

#numpy.squeeze(numpy.log(dataset.whitened_spectrogram[0:256,0:128])).plot(vmin=-10, vmax=5)
pyplot.figure(figsize=[10,20])
numpy.log(dataset.whitened_spectrogram[0:256,0:128]).plot(vmin=-10, vmax=5)
pyplot.show()

#dataset.whitened_spectrogram[0:256,0:128].plot(norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))

#matplotlib.style.use('ggplot')
#dataset.whitened_spectrogram.plot(logy=True, x_compat=True)
#matplotlib gcf().autofmt_xdate()

# dataset.notnull
# dataset.isnull

threshold = numpy.percentile(dataset.whitened_spectrogram, 95.0, axis=dataset.whitened_spectrogram.get_axis_num('GPS_time'))
threshold = numpy.percentile(dataset.whitened_spectrogram, 95.0)
# TODO chiedere valore a Pia
# TODO un unico istogramma per tutti, con un unico valore di soglia? oppure valore running nel tempo?

# per una gaussiana, i valori oltre 2.5 sigma sono circa lo 0.5%
# TODO quindi per ora scelgo il percentile di 99.5
#threshold = numpy.percentile(dataset.whitened_spectrogram, 99.5)

peakmap = numpy.greater_equal(dataset.whitened_spectrogram, threshold) # binary representation
peakmap.name = 'peakmap'
# TODO valutare le matrici sparse


# TODO cercare i massimi locali in un'immagine rumorosa, evitando la rumorosissima derivata prima discreta


numpy.squeeze(peakmap[0:256,0:128]).plot(cmap='gray_r'); pyplot.show()
# TODO vedere perché sembrano esserci un sacco di buchi (NaN) in dataset.whitened_spectrogram
# numpy.squeeze(numpy.log(dataset.whitened_spectrogram)).plot(); pyplot.show()

# TODO l'errore della corruzione sulle linee credo sia nella funzione safe_logarithm
#plot_RGB_image(safe_logarithm(selected_power_spectrum)[0:256,0:128], kwargs={'cmap':'bwr'})
# e anche in log_normalize

#log_values[numpy.isinf(log_values)] = 0
#ValueError: orthogonal array indexing only supports 1d arrays


# TODO vedere libreria python fatta da quelli di LIGO

    peakmap e massimi locali forse rumorosi
    sqrt(2.5) non in log (mediana=1) (MA non sotto radice perché uso gli spettri al quadrato)
    numpy.log(2.5) = 0.9162 (mettere riferimenti ad articoli)
    lo sbiancamento non è gaussiano, è solo bianco. non ci sono 'sigma'
    solo massimi locali
    segnale vero stretto, disturbo largo. i picchi stretti (CW) sopravvivono allo sbiancamento. dunque si può anche iniettare il segnale dopo lo sbiancamento
    estrarre parametri e durata del segnale e localizzazione in frequenza per darlo alla Hough gerarchica (per massimizzare il suo rapporto segnale/rumore)
    confronto sotto soglia della Hough
    la Hough è solo per il bianco e il nero: non funziona per i toni di grigio
    segnale transiente dal merger di stella di neutroni
    whitening con R 'ratio'        




