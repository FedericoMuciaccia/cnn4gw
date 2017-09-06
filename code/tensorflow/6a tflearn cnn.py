
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


# NOTA: un ragazzo che studiava medicina preso a caso in biblioteca ci ha messo <60 immagini per capire come classificare

import xarray

# data loading
train_images = xarray.open_dataarray('/storage/users/Muciaccia/train_images.netCDF4')
train_classes = xarray.open_dataarray('/storage/users/Muciaccia/train_classes.netCDF4')
test_images = xarray.open_dataarray('/storage/users/Muciaccia/test_images.netCDF4')
test_classes = xarray.open_dataarray('/storage/users/Muciaccia/test_classes.netCDF4')

number_of_train_samples, height, width, channels = train_images.shape
number_of_train_samples, number_of_classes = train_classes.shape

import tflearn

# build the convolutional network
network = tflearn.layers.core.input_data(shape=[None, height, width, channels], name='input')
#network = tflearn.layers.core.dropout(network, 0.8)
for i in range(6): # 6 convolutional block is the maximum dept with the given image size
    network = tflearn.layers.conv.conv_2d(network, nb_filter=9, filter_size=3, strides=1, padding='valid', activation='linear', bias=True, weights_init='uniform_scaling', bias_init='zeros', regularizer=None, weight_decay=0) # regularizer='L2', weight_decay=0.001, scope=None
    network = tflearn.layers.normalization.local_response_normalization(network)
    network = tflearn.activation(network, activation='relu')
    network = tflearn.layers.conv.max_pool_2d(network, kernel_size=2) # strides=None, padding='same'
    #network = tflearn.layers.normalization.local_response_normalization(network)
network = tflearn.layers.core.flatten(network)
#network = tflearn.layers.core.dropout(network, 0.8)
#network = tflearn.layers.core.fully_connected(network, n_units=10, activation='relu') # TODO regularizer and weight decay
network = tflearn.layers.core.fully_connected(network, n_units=number_of_classes, bias=True, activation='softmax', weights_init='truncated_normal', bias_init='zeros', regularizer=None, weight_decay=0)
#network = tflearn.layers.core.fully_connected(network, n_units=number_of_classes, bias=True, weights_init='truncated_normal', bias_init='zeros', activation='softmax') # weight_decay=0.001, scope=None
network = tflearn.layers.estimator.regression(network, optimizer='adam', learning_rate=0.001, batch_size=128, loss='categorical_crossentropy', shuffle_batches=True, name='target') # metric='default', to_one_hot=False, n_classes=None, validation_monitors=None

# TODO mettere solo un neurone finale, provare batch_size 64, controllare feed_dict nell'altra rete, provare ad aumentare il segnale (mettendolo a 1), controllare summary della rete, mettere normalizzazione al posto corretto, fare la prova in bianconero, controllare i valori strani di validation loss and accuracy

# df/dt = spindown
# 0-order pipeline
# Band Sample Data (BSD)
# manuale Sergio Frasca su BSD
# sfdb nel tempo levando il primo e l'iltimo quarto perché interallacciate
# frame grezzi (nel tempo) (fft e poi estrarre la banda e pulire tutto)
# libreria pubblica di LIGO lal-lalaps (Paola Laeci) (cercare eccessi di potenza per le wavelet)
# pss_frameS.c (per leggere i frame)
# f = 0 - 2048
# t_FFT circa come t_coerenza (per massimizzare il SNR)
# teamspeak + reveal.js per le slides sincronizzate
# vidyo (video+slides+group_chat)
# talk massimo 20 minuti
# numero DCC messo nella presentazione (ad una conferenza)

# pwd nella home
# /home/VIRGO/muciaccia
# software su virgo3
# dati su virgo4 (molto spazio)
# lcg-tools (grid) + directory speciale in virgo4 con scrittura normale
# path logico VS path fisico
# cartella magica: /storage/gpfs_virgo4/CW_Roma1/
# ui01 ui02 ui03 (poco potenti)
# referenti CNAF: Matteo Tenti, Lucia Morganti

# dati Virgo 02 (C01 ?)
# dati BSD nel tempo
# whitening
# limite peakmap
# macchina con GPU
# metodo per rigetto spettri e differenza relativa
# atoregressivo (articolo Pia) (codice pss_sfdb.c) short_psar_rev_freq (media autoregressiva fatta dal basso verso l'alto (ps spectrum ar autoregressive rev reversal freq frequency))
# whitening virgo
# lunghezza scritto 100+ pagine
# problemi col training
# in-memory
# .h5 file extension
# soglia di selezione fatta in base alla rilevazione sui segnali iniettati su tutti gli spettri indiscriminatamente (ottimizzare la ripulitura dei dati)

# NOTE e TODO:
# tra i miei 10 minuti di training e la settimana di traning scritta nell'articolo c'è una differenza esattamente di un fattore mille
# poi il numero di kernel andrà ottimizzato guardando le immagini generate massimizzando il gradiente, in modo da essere sicuri di non star levando spazio a features rilevanti (come ad esempio tutte le varie combinazioni di colore)
# vedere se si riesce ad andare sotto la soglia della peakmap, nel qual caso si può alaborare una strategia per analizzare anche i segnali continui
# nel futuro fa estrarre alla rete anche i parametri del segnale (o farlo fare ad una rete dedicata, a valle di una pulizia ottimale del segnale)
# nel futuro far fare la selezione degli spettri direttamente ad un sistema automatico
# l'eventuale dropout iniziale di fatto gioca il ruolo di data augmentation
# in futuro farlo direttamente con gli streaming di dati che escono dall'interferometro
# ogni tanto il training non ingrana per niente e bisogna spegnere e ricominciare da capo
# valutare max_pooling VS average_pooling
# studiare local_response_normalization


model = tflearn.DNN(network, tensorboard_verbose=0) # 3

class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def on_epoch_end(self, training_state):
        if training_state.acc_value > 0.9999: # when the training accuracy is 1 the model cannot learn further
            raise StopIteration

# load pretrained weights (to start closer to the minimum)
model.load('/storage/users/Muciaccia/models/pretraining_amplitude_5.tflearn')

# training
try:
    model.fit({'input':train_images}, {'target':train_classes}, n_epoch=100, validation_set=({'input':test_images}, {'target':test_classes}), snapshot_step=100, show_metric=True, callbacks=EarlyStoppingCallback()) # run_id='tflearn_conv_net_trial'
except StopIteration:
    print('train accuracy is 1: training finished!')

# save the model
model.save('/storage/users/Muciaccia/models/pretraining_amplitude_1.tflearn')

# tempo pretraining con segnale a 10: 4 minuti, <30 epoche con 2500 immagini di train
# tempo pretraining con segnale a 5: 1 minuto, <10 epoche

# TODO:
# predizioni, grafici loss e accuracy/error, confusion_matrix
# istogrammi di classificazione (con signoide finale)
# veto sui dati iniziali
# iniezione distribuita in chunck per l'out-of-memory
# iniezione di segnali veri con fft (e loro istogramma per capirne il livello)
# script curriculum learning automatizzato (vedere model.load)
# calcolo complessivo SNR con l'energia
# generare corretta forma d'onda
# log10 nei grafici
# spettro autoregressivo nel grafico di whitening
# mediana, 50%, 90%, 100% nel grafico dello spettro completo
# valutare vari livelli di accuracy col classificatore finale
# cominciare a scrivere!
# leggere articolo
# documentarsi su curve ROC e su local_response_normalization
# script per fare 100 training e poter fare un grafico

