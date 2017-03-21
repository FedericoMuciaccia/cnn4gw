
# coding: utf-8

# In[3]:

import scipy.io
from matplotlib import pyplot
import pandas
import skimage.io
import numpy
import os

get_ipython().magic('matplotlib inline')


# In[4]:

# amplitude=0.03e-22
filename_visible_signal = './raw data/simulated signal on gaussian white noise background/0.03e-22/GREYfiles_64_SIGnum_029_.mat'

# amplitude=0.003e-22
filename_invisible_signal = './dati esclusi (TODO)/troppo piccoli (invisibili)/0.003e-22/GREYfiles_100_SIGnum_029_.mat'


# In[37]:

def load_and_plot(filename, amplitude):

    a = scipy.io.loadmat(filename)
    b = a['SUB_peaks']
    x = b[0,:] - b[0,1]
    y = b[1]
    #y = y[y>0]

    z = b[2]
    df = pandas.DataFrame({'time':x, 'frequency':y, 'significance':z})
    #df = pandas.DataFrame({'time':x, 'frequency':y})
    df = df[df.frequency > 0]

    x = df.time.values
    y = df.frequency.values

    # normalized significance
    #df.significance = (df.significance - df.significance.min())/(df.significance.max() - df.significance.min())
    ##gray_colors = df.significance.apply(str).values
    ##gray_colors = df.significance.astype('float16').apply(str).values
    z = df.significance.values
    ##pyplot.scatter(x,y, c=gray_colors)
    
    # la soglia è messa a 2.5
    
    #print(a['WHEREIS'])
    limiti_segnale = numpy.around(a['WHEREIS'].flatten(), decimals=2).tolist()
    
    pyplot.figure(figsize=(10,6))
    
    pyplot.title("Significance above 2.5 threshold \n (signal amplitude = {0}, signal interval = {1})".format(amplitude, limiti_segnale))
    pyplot.xlabel('Days from the beginning')
    pyplot.ylabel('Frequency [Hz]')
    pyplot.xlim(min(x), max(x))
    pyplot.ylim(min(y), max(y))
    # pyplot.scatter(x, y, s=z)
    pyplot.scatter(x, y, c=z, cmap='gray_r', marker='.', linewidth=0)
    pyplot.colorbar()
    
    pyplot.savefig(amplitude + '_signal_significance.jpg')
    pyplot.close()


# In[40]:

load_and_plot(filename_visible_signal, amplitude='0.03e-22')


# In[41]:

load_and_plot(filename_invisible_signal, amplitude='0.003e-22')


# In[ ]:



