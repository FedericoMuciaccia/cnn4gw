
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
import tensorflow as tf

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
signal_spindown = delta_frequency/signal_duration # = df/dt

time_sampling_rate = 1#4096 # Hz
time_resolution = 1/time_sampling_rate

time_bin = 8192 # t_FFT

image_time_start = 0.0
image_time_interval = 2**19 # 2^19 = 524288 seconds (little more than 6 days)
image_time_stop = image_time_start + image_time_interval

t = numpy.arange(start=image_time_start, stop=image_time_stop, step=time_resolution, dtype=numpy.float32) # TODO lentissimo
#t = tf.linspace(start=image_time_start, stop=image_time_stop, num=image_time_interval*time_sampling_rate + 1) # last value included
# TODO controllare troncamenti tempo in float32

noise_amplitude = 1e-6 # TODO check
real_part, imaginary_part = noise_amplitude*numpy.random.randn(2,len(t)).astype(numpy.float32)
white_noise = real_part + 1j*imaginary_part
#real_part = tf.random_normal(shape=t.shape, mean=0, stddev=noise_amplitude) # dtype=float32
#imaginary_part = tf.random_normal(shape=t.shape, mean=0, stddev=noise_amplitude) # dtype=float32
#white_noise = tf.complex(real_part, imaginary_part) # dtype=complex64

# TODO BUG: tf.random_normal mu and sigma
# TODO BUG: range(start=0, stop, step=2) VS ordered dictionary
# TODO VEDERE tf.shape di array 1D e di scalari e obblico di mettere sempre le parentesi quadre negli argomenti shape delle funzioni

signal_amplitude = 1*noise_amplitude # SNR ? # ratio=0.1 ancora ben visibile
#omega = 2*pi*frequency
signal = signal_amplitude*numpy.exp(1j*2*numpy.pi*(signal_starting_frequency - signal_spindown*t)*t)
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

# TODO unilatera e bilatera
# TODO parallelizzare il calcolo sui vari chunks
fft_data = numpy.array(list(map(numpy.fft.fft, chunks))).astype(numpy.complex64)
# TODO vedere tipo di finestra
spectra = numpy.square(numpy.abs(fft_data)) # TODO sqrt(2) etc etc
spectrogram = numpy.transpose(spectra)[0:256]
whitened_spectrogram = spectrogram/numpy.median(spectrogram) # 5.7280793e-09

pyplot.figure(figsize=[15,10])
pyplot.hist(numpy.log10(spectrogram.flatten()), bins=300) # log10
pyplot.show()
# TODO valutarre la sovrapponibilità coi dati veri

pyplot.figure(figsize=[15,10])
pyplot.hist(numpy.log10(whitened_spectrogram.flatten()), bins=300) #log10
pyplot.vlines(x=numpy.log10(2.5), ymin=0, ymax=700, color='orange', label='peakmap threshold = 2.5')
linear_ticks = numpy.linspace(-5, 5, num=11)
log_labels = ['10^{}'.format(int(i)) for i in linear_ticks]
pyplot.xticks(linear_ticks, log_labels)
pyplot.legend()
pyplot.show()

# NOTE: FFT (Fast Fourier Transform) refers to a way the discrete Fourier Transform (DFT) can be calculated efficiently, by using symmetries in the calculated terms. The symmetry is highest when `n` is a power of 2, and the transform is therefore most efficient for these sizes.

pyplot.figure(figsize=[10,10*256/148])
pyplot.imshow(numpy.log(whitened_spectrogram), origin="lower", interpolation="none", cmap='gray')
pyplot.show()






