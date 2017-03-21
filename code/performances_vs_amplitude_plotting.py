
# Copyright (C) 2016  Federico Muciaccia (federicomuciaccia@gmail.com)
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
import pandas

import matplotlib
matplotlib.use('SVG')
from matplotlib import pyplot

performances = pandas.read_csv('./media/performances_vs_amplitude.csv')
# amplitude, precision/purity, recall/efficiency


possible_amplitudes = numpy.array([0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005]) * 1e-22


fig, [ax1, ax3] = pyplot.subplots(2, sharex=True, figsize=(8,6))
fig.suptitle('model validation performances', size=12)
ax1.plot(performances.amplitude, performances.efficiency, color='#3366ff', lw=2)
ax1.set_ylabel('efficiency')
ax1.set_xlim((0.1e-22,0.005e-22))
ax1.set_ylim((0.6,1.0))
#ax1.set_yscale('log')
ax3.plot(performances.amplitude, performances.purity, color='#3366ff', lw=2)
ax3.set_ylabel('purity')
ax3.set_xlabel('amplitude')
ax3.set_ylim((0.9,1.0))
#ax3.set_yscale('log')
ax3.set_xscale('log')
#pyplot.show()
fig.savefig('./media/performances_vs_amplitude.svg')
pyplot.close()


