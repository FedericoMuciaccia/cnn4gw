
import numpy
import xarray

H_dataset = xarray.open_mfdataset('/storage/users/Muciaccia/netCDF4/O2/C01/128Hz/LIGO Hanford*.netCDF4')

# TODO per ora generare un dataset fittizio per Virgo. poi prenderlo dai nuovi dati di O2 o dai vecchi dati di VSR4
V_dataset = H_dataset.copy()
V_dataset.update({'detector':['Virgo']}) # TODO inplace
#RGB_median = 3e-07, 2e-07, 2.5e-07
V_spectrogram = 2.5e-7 * numpy.ones_like(V_dataset.spectrogram)
V_dataset.update({'spectrogram':(['frequency','GPS_time','detector'],V_spectrogram)})
#del V_spectrogram
V_locally_science_ready = numpy.ones_like(V_dataset.locally_science_ready).astype(bool) # all True
V_dataset.update({'locally_science_ready':(['GPS_time','detector'],V_locally_science_ready)})

V_dataset.to_netcdf('/storage/users/Muciaccia/netCDF4/fake_Virgo_dataset.netCDF4', format='NETCDF4')
