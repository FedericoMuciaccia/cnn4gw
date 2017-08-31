
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


import glob
import xarray
import scipy.io

def convert_file(file_path):
    dataset = xarray.open_dataset(file_path)
    file_name = 'amplitude_{}.mat'.format(dataset.signal_intensity)
    scipy.io.savemat('/storage/users/Muciaccia/confronto_Hough/mat/'+file_name,
                     mdict={'H':dataset.images.sel(channels='red').values,
                            'L':dataset.images.sel(channels='green').values,
                            'V':dataset.images.sel(channels='blue').values,
                            'signal_amplitude':dataset.signal_intensity})

file_list = glob.glob('/storage/users/Muciaccia/confronto_Hough/netCDF4/*.netCDF4', recursive=True)

for file_path in file_list:
    convert_file(file_path)

