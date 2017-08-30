import glob
import xarray
import scipy.io

def convert_file(file_path):
    dataset = xarray.open_dataset(file_path)
    file_name = 'amplitude_{}.mat'.format(dataset.signal_intensity)
    scipy.io.savemat('/storage/users/Muciaccia/confronto_Hough/mat/'+file_name, mdict={'H':dataset.images.sel(channels='red').values,
                                                   'L':dataset.images.sel(channels='green').values,
                                                   'V':dataset.images.sel(channels='blue').values,
                                                   'signal_amplitude':dataset.signal_intensity})

file_list = glob.glob('/storage/users/Muciaccia/confronto_Hough/netCDF4/*.netCDF4', recursive=True)

for file_path in file_list:
    convert_file(file_path)

