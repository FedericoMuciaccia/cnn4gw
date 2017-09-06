
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
#import tensorflow as tf

from matplotlib import pyplot

second = 1
minute = 60*second
hour = 60*minute
day = 24*hour

signal_time_start = 2*day
signal_duration = 2*day
signal_time_stop = signal_time_start + signal_duration

signal_starting_frequency = 10#90 # Hz
delta_frequency = -0.005 # Hz
signal_spindown = delta_frequency/signal_duration # = df/dt # -10^-9 (piccolo) # -10^-8 (grandetto) # binning_spindown = delta_f/t_durata_segnale

time_sampling_rate = 1#4096 # Hz # lì 256 Hz sottocampionati
time_resolution = 1/time_sampling_rate

time_bin = 8192 # t_FFT

image_time_start = 0.0
image_time_interval = 2**19 # 2^19 = 524288 seconds (little more than 6 days)
image_time_stop = image_time_start + image_time_interval

t = numpy.arange(start=image_time_start, stop=image_time_stop, step=time_resolution, dtype=numpy.float32) # TODO lentissimo
#t = tf.linspace(start=image_time_start, stop=image_time_stop, num=image_time_interval*time_sampling_rate + 1) # last value included
# TODO controllare troncamenti tempo in float32

noise_amplitude = 1.5e-5 # --> 1.3e-6 # TODO check normalizzazione
# 5e-4 --> 1.5e-3
# 1e-4 --> 5.5e-5
# 5e-5 --> 1.4e-5
# 2.5e-5 --> 3.6e-6
# 2e-5 --> 2.3e-6
# 1.5e-5 --> 1.3e-6
# 1e-5 --> 5.5e-7
# 1e-6 --> 5.5e-9
# 
# ???? --> 1e-6
real_part, imaginary_part = noise_amplitude*numpy.random.randn(2,len(t)).astype(numpy.float32)
white_noise = real_part + 1j*imaginary_part
#real_part = tf.random_normal(shape=t.shape, mean=0, stddev=noise_amplitude) # dtype=float32
#imaginary_part = tf.random_normal(shape=t.shape, mean=0, stddev=noise_amplitude) # dtype=float32
#white_noise = tf.complex(real_part, imaginary_part) # dtype=complex64

# TODO BUG: tf.random_normal mu and sigma
# TODO BUG: range(start=0, stop, step=2) VS ordered dictionary
# TODO VEDERE tf.shape di array 1D e di scalari e obblico di mettere sempre le parentesi quadre negli argomenti shape delle funzioni e dimensioni con label/nome tipo xarray e pandas


def signal_waveform(t):
    # signal = exp(i phi(t))
    # phi(t) = integrate_0^t{omega(tau) d_tau}
    # omega = 2*pi*frequency
    # df/dt = s # linear (first order) spindown
    # f(t) = f_0 + s*t
    # ==> phi(t) = integrate_0^t{2*pi*(f_0+s*tau) d_tau}
    # ==> phi(t) = 2*pi*(f_0*t + (1/2)*s*t^2 + C) # TODO capire perché è necessario mettere 'modulo 2 pi'
    return signal_amplitude*numpy.exp(1j*numpy.mod((2*numpy.pi*(signal_starting_frequency + (1/2)*signal_spindown*t)*t),2*numpy.pi))

signal_amplitude = 1*noise_amplitude # SNR in un solo chunk di dati ? # ratio=0.1 ancora ben visibile # rapporto critico = (amp segnale - media del rumore)/deviazione standard = 1 # poi scaling con CR*sqrt(N_FFT) # N_FFT = numero di chunk temporali # rapporto critico nel piano della Hough (coi conteggi in quel piano) # calcolare quanta è l'energia trasportata dal segnale (integrale dello spettro di potenza?) (VS valore di picco della sinusoide) # energia totale (integrale) ceduta dal segnale nel rivelatore # vs SNS_su_singola_FFT # l'ampiezza della FFT diminuisce col tempo, perché ci sono meno cicli nel tempo dato che la frequenza diminuisce # procedura completamente coerente, con un'unica FFT (coerente e incoerente, con o senza sqrt(durata segnale OR t_osservazione) (vedere articolo Explorer) # signal power density
signal = signal_waveform(t)
#signal = signal_amplitude*tf.exp(tf.complex(1.0,2*numpy.pi*(signal_starting_frequency - signal_spindown*t)*t))

signal[numpy.logical_or(t < signal_time_start, t > signal_time_stop)] = 0
# TODO farlo con le slice su xarray

pyplot.figure(figsize=[15,10])
pyplot.plot(numpy.real(white_noise)) # TODO senza logy?
pyplot.plot(numpy.real(signal))
pyplot.show()

pyplot.figure(figsize=[15,10])
pyplot.title('injected signal and gaussian white noise')
pyplot.plot(numpy.real(white_noise[300000:300500])) # TODO senza logy?
pyplot.plot(numpy.real(signal[300000:300500]))
pyplot.show()

pyplot.figure(figsize=[15,10])
pyplot.plot(numpy.real(white_noise+signal)) # TODO senza logy?
pyplot.show()

number_of_chunks = len(t)/(time_bin/2) # TODO hack per le interallacciate
chunks = numpy.split(white_noise+signal, number_of_chunks)
#chunks = numpy.split(signal, number_of_chunks)

pyplot.figure(figsize=[15,10])
pyplot.hist2d(t, numpy.real(white_noise), bins=[100,number_of_chunks])
pyplot.show()

pyplot.figure(figsize=[15,10])
pyplot.hist2d(t, numpy.real(white_noise+signal), bins=[100,number_of_chunks])
pyplot.show()


# TODO parallelizzare il calcolo sui vari chunks
fft_data = numpy.array(list(map(numpy.fft.fft, chunks))).astype(numpy.complex64)
# TODO unilatera (rfft) e bilatera (fft)
# TODO rimettere ordine corretto nel caso complesso e shiftare lo zero
# TODO vedere tipo di finestra e interlacciatura e normalizzazione per la potenza persa
spectra = numpy.square(numpy.abs(fft_data)) # TODO sqrt(2) etc etc
# TODO normd (normalizzare sul numero di dati)
spectrogram = numpy.transpose(spectra)[0:256] # TODO hack forse sbagliato
whitened_spectrogram = spectrogram/numpy.median(spectrogram)

# power_spectrum = FFT autocorrelazione
# power_spectrum != modulo quadro dell'FFT (serve un fattore di normalizzazione)
# normd circa 10^-5 o 10^-6
# dati reali con livello a 10^-23
# simulare rumore bianco in frequenza direttamente con una gaussiana di larghezza 1/sigma
# finestra flat_coseno per minimizzare l'allargamento di segnali che variano un poco in frequenza (per smussare i bordi). e dunque c'è poi la necessità di buttare i bordi mediante le FFt interallacciate. (minimizzare i ghost laterali della delta di Dirac allargata della sinusoide e/o massimizzare l'altezza del picco). poi usare normw per tenere in conto della potenza persa ai bordi della finestra rispetto alla funzione gradino (fattore comuque vicino ad 1). tutti fattori da rimoltiplicare per controbilanciare la perdita di potenza spettrale


pyplot.figure(figsize=[15,10])
spectrum_median = numpy.median(spectra[0]) # deve fare circa 1e-6
pyplot.hlines(y=spectrum_median, xmin=0, xmax=4096, color='black', label='spectrum median = {}'.format(spectrum_median), zorder=1) # TODO invece che la mediana plottare lo spettro autoregressivo
pyplot.semilogy(spectra[0], label='raw spectrum', zorder=0) # grafico obbligatoriamente col logaritmo in y
pyplot.title('frequency spectrum')
pyplot.xlabel('frequency') # Hz OR index ?
pyplot.legend(loc='lower right', frameon=False)
pyplot.show()
# TODO WISHLIST: spectrogram[all,1]

pyplot.figure(figsize=[15,10])
pyplot.hist(numpy.log10(spectrogram.flatten()), bins=300) # log10
pyplot.show()
# TODO valutarre la sovrapponibilità coi dati veri

pyplot.figure(figsize=[15,10])
pyplot.hist(numpy.log10(whitened_spectrogram.flatten()), bins=300) #log10
pyplot.vlines(x=numpy.log10(2.5), ymin=0, ymax=700, color='orange', label='peakmap threshold = 2.5') # sqrt(2.5)
linear_ticks = numpy.linspace(-5, 5, num=11)
log_labels = ['10^{}'.format(int(i)) for i in linear_ticks]
pyplot.xticks(linear_ticks, log_labels)
pyplot.legend()
pyplot.show()

# NOTE: FFT (Fast Fourier Transform) refers to a way the discrete Fourier Transform (DFT) can be calculated efficiently, by using symmetries in the calculated terms. The symmetry is highest when `n` is a power of 2, and the transform is therefore most efficient for these sizes.

pyplot.figure(figsize=[10,10*256/148])
pyplot.imshow(numpy.log(whitened_spectrogram), origin="lower", interpolation="none", cmap='gray')
pyplot.show()

# NOTA: trattare sempre grandezze relative e pubblicare grafici e risultati con i quali tutti i diversi esperimenti possono confrontarsi indipendentemente

# domande Pia:
# forma d'onda con funzione analitica precisa
# sembrano più sinusoidi modulate
# ghosts dovuti alla finestra?
# Hamming window
# spindown a salire ed inversione di fase
# spindown lineare vs peakmap con onda continua ondeggiante
# livello ampiezza rumore bianco per sovrapponibilità coi dati veri
# fattore di normalizzazione per gli spettri (radice di 2 etc)
# unilatero e bilatero
# fft con sliding gaussiano invece che interallacciate
# FFT(real) = real ?
# campionamento originale 4096 Hz
# tipici valori di spindown
# SNR 1 vs 1 inverosimile. più ragionevole 1 vs 0.1
# varie unità di misura nei grafici

# fft reale bilatera con ghost (dividendo l'energia/potenza)
# fft complessa senza frequenze negative
# campionare a frequenza doppia
# i dati veri sono reali e non complessi
# iniettare solo in una piccola banda (nei complessi) per velocizzare il calcolo
# iniettare nel tempo per avere tutti gli artefatti
# ipoteticamente, dato che 128 pixel temporali corrispondono a circa 6 giorni, si potrebbe replicare l'analisi con segnali continui di 3 mesi, con 2048 pixel temporali (e ovviamente un batch-size piccolissimo)

# flat cos flat top (Sergio)
# integrale con montecarlo
# niente punti del cielo e linee divergenti per le varie correzioni Doppler e linee discontinue per i pattern d'antenna
# [punto_del_cielo, frequenza_base, vari_ordini_di_spindown]
# gd->y sono i dati (struttura)
# interlacciamento
# documento che descrive la forma della finestra e il suo perché
# loop fatto al contrario
# normw = 2*sqrt(2/3) calcolato numerico o integrando simbolicamente
# pipeline: data_stream, trigger, denoiser, nonparametric_fit, parameter_extractor
# chiedere simulazione reale per i pattern delle linee divergenti che si vedrebbero (Doppler + pattern d'antenna)
# generare il segnale nel tempo con tensorflow
# data_generator/queue per i vari file in-memory con tensorflow
# data stream direttamete da LIGO, non dal CNAF
# running window
# generare segnali a parte e poi fare funzione add_signal([noise_image, signal]) che tenga conto dei buchi dell'immagine
# usare tensorflow e SparseTensor per generare segnali in blocco
# vedere vectorialization su tensorflow con tf.map_fn (BUG su numpy)
# serialization on tensorflow: define a queue -> define single preprocessing -> set a batch to preprocess
# fare istogrammi puliti con fft del segnale
# denoiser a cui si danno in pasto add_signal(rumore,segnale) e solo segnale (eventualemte senza i buchi) come target

# 
# randn su GPU
# out-of-core con massimo 8 GB di RAM (memoria GPU)





def flat_top_cosine_edge_window(data_chunk, window_lenght = 8192):
    # 'flat top cosine edge' window function (by Sergio Frasca)
    # structure: [ascending_cosine, flat, flat, descending_cosine]

    half_lenght = int(window_lenght/2)
    quarter_lenght = int(window_lenght/4)
    
    index = numpy.arange(window_lenght)
        
    # sinusoidal part at the edges
    factor = 0.5 - 0.5*numpy.cos(2*numpy.pi*index/half_lenght)
    # flat part in the middle
    factor[quarter_lenght:window_lenght-quarter_lenght] = 1
    # TODO plottare la funzione finestra e la sua trasformata di Fourier
    
    # TODO attenzione all'ultimo valore:
    # factor[8191] non è 0
    # (perché dovrebbe esserlo invece factor[8192], ma che è fuori range)
    
    # calcolo delle normalizzazione necessaria per tenere in conto della potenza persa nella finestra
    # area sotto la curva diviso area del rettangolo totale
    # se facciamo una operazione di scala (tanto il rapporto è invariante) si capisce meglio
    # rettangolo: [x da 0 a 2*pi, y da 0 a 1]
    # area sotto il seno equivalente all'integrale del seno da 0 a pi
    # integrate_0^pi sin(x) dx = -cos(x)|^pi_0 = 2
    # area sotto il flat top: pi*1
    # dunque area totale sotto la finestra = 2+pi
    # area del rettangolo complessivo = 2*pi*1
    # potenza persa (rapporto) = (2+pi)/2*pi = 1/pi + 1/2 = 0.818310
    # fattore di riscalamento = 1/potenza_persa = 1.222031
    # TODO questo cacolo è corretto? nel loro codice sembra esserci un integrale numerico sui quadrati
    # caso coi quadrati:
    # integrate sin^2 from 0 to pi = x/2 - (1/4)*sin(2*x) |^pi_0 = 
    # = pi/2
    # dunque (pi/2 + pi)/2*pi = 3/4
    
    pyplot.figure(figsize=[15,10])
    pyplot.plot(factor)
    pyplot.show()
    
    factor = index.copy()
    factor[4096:8192] = 8192-index[4096:8192]
    
    #factor = numpy.random.randn(8192)
    
    x = numpy.linspace(0, 2*numpy.pi, 8192)
    
    noise = 0.01*numpy.random.randn(8192)
    y1 = numpy.sin(20*x)
    y2 = 2*numpy.sin(40*x)
    y = y1 + y2 # + noise
    
    y = factor + noise
    
    pyplot.figure(figsize=[15,10])
    #pyplot.plot(x, noise)
    pyplot.plot(x, y1)
    pyplot.plot(x, y2)
    pyplot.plot(x, y)
    pyplot.show()
    
    pyplot.figure(figsize=[15,10])
    pyplot.plot(y)
    pyplot.show()

    window_fft = numpy.real(numpy.fft.fft(y))
    # TODO con fft (bilatera) fare attenzione a rimettere le due metà ordinate nello spettro e a centrare tutto correttamente
    # TODO con rfft (unilatera) attenzione alle potenze di 2
    
    fft = numpy.fft.fft(y)
    dBV = 20*numpy.log(numpy.sqrt(numpy.square(numpy.abs(fft)))/8192)
    
    pyplot.figure(figsize=[15,10])
    pyplot.plot(dBV)
    pyplot.show()
    
    pyplot.figure(figsize=[15,10])
    pyplot.semilogy(x, window_fft)
    pyplot.show()
    
    pyplot.figure(figsize=[15,10])
    pyplot.plot(window_fft[0:50])
    pyplot.show()
    
    data_chunk = data_chunk*factor
    
    return data_chunk









