
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

from matplotlib import pyplot

second = 1
minute = 60*second
hour = 60*minute
day = 24*hour

t_start = 2.5*day
duration = 2*day
t_end = t_start + duration

starting_frequency = 10#90 # Hz
#omega = 2*numpy.pi*frequency
delta_frequency = -0.005 # Hz
spindown = delta_frequency/duration

time_sampling_rate = 1 #4096 # Hz
time_resolution = 1/time_sampling_rate
total_time_duration = 73*time_bin #7*day # TODO per far quadrare i conti
t = numpy.arange(start=0, stop=total_time_duration, step=time_resolution, dtype=numpy.float32) # TODO lentissimo

noise_amplitude = 1e-6 # TODO check
real_part = noise_amplitude*numpy.random.randn(len(t)).astype(numpy.float32)
imaginary_part = 1j*noise_amplitude*numpy.random.randn(len(t)).astype(numpy.float32)
white_noise = real_part + imaginary_part

signal_amplitude = 1*noise_amplitude
signal = signal_amplitude*numpy.exp(1j*2*numpy.pi*(starting_frequency - spindown*t)*t)

signal[numpy.logical_not(numpy.logical_and(t_start < t, t < t_end))] = 0
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

time_bin = 8192 # s # TODO fare in modo che combacino con la fine dell'immagine
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

pyplot.hist(numpy.log(whitened_spectrogram.flatten()), bins=100)
pyplot.show()
# TODO dov'è la soglia?

# NOTE: FFT (Fast Fourier Transform) refers to a way the discrete Fourier Transform (DFT) can be calculated efficiently, by using symmetries in the calculated terms. The symmetry is highest when `n` is a power of 2, and the transform is therefore most efficient for these sizes.

pyplot.figure(figsize=[10,10*256/148])
pyplot.imshow(numpy.log(whitened_spectrogram), origin="lower", interpolation="none", cmap='gray')
pyplot.show()







