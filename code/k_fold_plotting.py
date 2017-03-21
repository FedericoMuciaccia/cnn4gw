
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

import matplotlib
matplotlib.use('SVG')
from matplotlib import pyplot

total_history = numpy.load('./images/k_fold history.npy')

# alphabetical order (axis=2)
names = ['acc','loss','val_acc','val_loss']

start_acc = 0.5 # accuracy nel caso di random guess a maximun entropy
start_loss = numpy.log(2) # misurando in base 2 si dovrebbe invece 

start_values = numpy.array([start_acc, start_loss,start_acc, start_loss])

start = numpy.zeros((10,1,4))
start[:] = start_values

train_history = numpy.concatenate([start, total_history], axis=1)

# classification error = 1 - accuracy
train_history[:,:,0] = 1 - train_history[:,:,0]
train_history[:,:,2] = 1 - train_history[:,:,2]

k_fold_number = 10

epochs = 50

means = train_history.mean(axis=0)

sigmas = train_history.std(axis=0)

def plot_errorband(axis, means, sigmas, color='blue'):
	x = numpy.arange(len(means))
	y_high = means + sigmas
	y_low = means - sigmas
	axis.fill_between(x, y_high, y_low, facecolor=color, alpha=0.1)

# TODO molte parti duplicate
fig, [ax1, ax3] = pyplot.subplots(2, sharex=True, figsize=(8,6))
fig.suptitle('model performances (k-fold with k=10)', size=12)
# train 1 - accuracy
ax1.plot(means[:,0], label='train', color='blue')
plot_errorband(ax1, means[:,0], sigmas[:,0], color='blue')
# test 1 - accuracy
ax1.plot(means[:,2], label='test', color='green')
plot_errorband(ax1, means[:,2], sigmas[:,2], color='green')
ax1.set_ylabel('classification error') # r'$1-$accuracy' = 'error'
ax1.legend(loc='best', frameon=False)
ax1.set_xlim((0,50))
ax1.set_ylim((1e-3,1e-0))
ax1.set_yscale('log')
#sigma_levels = [0.6826895, 0.9544997, 0.9973002, 0.9999366, 0.9999994]
#sigma_levels = numpy.array(sigma_levels)
#ax1.hlines(1-sigma_levels, *ax1.get_xlim(), linestyles='dotted', alpha=0.5)#, color='red')
#ax2 = ax1.twinx()
#ax2.set_yscale('log')
#ax2.set_ylim(ax1.get_ylim())
#ax2.set_yticks(1-sigma_levels)
#ax2.set_yticklabels([r'$1 \sigma$', r'$2 \sigma$', r'$3 \sigma$', r'$4 \sigma$', r'$5 \sigma$'])#, ha='left')
#ax2.minorticks_off()
# train loss
ax3.plot(means[:,1], label='train', color='blue')
plot_errorband(ax3, means[:,1], sigmas[:,1], color='blue')
# test loss
ax3.plot(means[:,3], label='test', color='green')
plot_errorband(ax3, means[:,3], sigmas[:,3], color='green')
ax3.set_ylabel('loss')
ax3.set_xlabel('epoch')
ax3.legend(loc='best', frameon=False)
ax3.set_ylim((1e-2,1e-0))
ax3.set_yscale('log')
#pyplot.show()
fig.savefig('./images/k_fold_training_history.svg')
pyplot.close()


