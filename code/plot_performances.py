
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
# backend per poter lavorare in remoto senza il server X
# PRIMA di importare pyplot
matplotlib.use('SVG')
from matplotlib import pyplot

from keras.models import load_model

from sklearn.metrics import confusion_matrix
#import sklearn.manifold

# import the model
model = load_model('./models/trained_model.h5')

# print the model summary on file
import sys
old_sddout = sys.stdout
my_file = open('./models/model_summary.txt', 'w+')
sys.stdout = my_file
model.summary()
my_file.close()
sys.stdout = old_sddout

# import the datasets
#train_images = numpy.load('./clean data/train_images (all).npy')
#train_classes = numpy.load('./clean data/train_classes (all).npy')
validation_images = numpy.load('./clean data/validation_images (all).npy')
validation_classes = numpy.load('./clean data/validation_classes (all).npy')


## plot class separability/problem hardness
## project data to two dimensions using t-SNE
## TODO dovrebbe preservare distanza tra i punti e topologia (CHECK)
#tsne = sklearn.manifold.TSNE() # TODO capire cos'è
#Xtrn = train_images.reshape(1200, 98*82*1) # shape (n_samples, n_features) # TODO generalizzare
## TODO è possibile dargli delle features più sensate?
#ytrn = train_classes
#X_2D = tsne.fit_transform(Xtrn) # TODO algoritmo NON parallelo!
#pyplot.figure()
#pyplot.title('t-SNE train data 2D projection')
##pyplot.grid()
#pyplot.xlim(-30, 30)
#pyplot.ylim(-30, 30)
##pyplot.xticks([])
##pyplot.yticks([])
#pyplot.scatter(X_2D[ytrn==0, 0], X_2D[ytrn==0, 1], c='red', alpha=0.5, marker='o', linewidth=0)
#pyplot.scatter(X_2D[ytrn==1, 0], X_2D[ytrn==1, 1], c='blue', alpha=0.5, marker='o', linewidth=0)
##pyplot.show()
#pyplot.savefig('separability.svg')
#pyplot.close()





# compute predictions
true_classes = validation_classes
predicted_classes = model.predict(validation_images)
predicted_classes = predicted_classes.reshape(true_classes.shape)
# TODO predict_classes()?
# TODO predict_proba()?
# TODO batch_size?


numpy_epsilon = numpy.finfo(numpy.float32).eps

@numpy.vectorize
def numpy_binary_crossentropy(predicted, true):
	#true[true == 0] = numpy_epsilon
	#true[true == 1] = 1 - numpy_epsilon
	#predicted[predicted == 0] = numpy_epsilon
	#predicted[predicted == 1] = 1 - numpy_epsilon
	# TODO oppure mettere il valore correttivo di epsilon direttamente nella formula
	return - (true * numpy.log(predicted + numpy_epsilon)) - ((1 - true) * numpy.log(1 - predicted + numpy_epsilon))

# TODO farlo anche con la categorical cross entropy
losses = numpy_binary_crossentropy(predicted_classes, true_classes)
pyplot.hist(losses, bins=50)
pyplot.title('final binary cross-entropy distribution')
pyplot.savefig('./media/final_loss_distribution.svg')
pyplot.close()

# TODO creare una custom metrics in keras per plottare la distribuzione dei loss ad ogni epoca, per vedere come evolve

#N = 100
#x = numpy.linspace(0, 1, N)
#y = numpy.linspace(0, 1, N)
#xx, yy = numpy.meshgrid(x, y)
#z = numpy_binary_crossentropy(xx, yy)
#
#pyplot.pcolor(xx, yy, z, cmap='viridis')
#pyplot.colorbar()
#pyplot.savefig('binary_crossentropy.svg')
#pyplot.close()



# TODO facendo la prova che effettivamente la rete abbia imparato correttamente tutti gli esempi, ovvero facendole fare le previsioni sul set di train, noto che non tutte le previsioni sono certe a 0 o 1, ma ci sono abbastanza esempi che ricadono intorno alla probabilità 0.5. come è possibile? come si può spiegare? come si fa a dire alla rete di classificare correttamente le cose che le dico con sicurezza a che classe appartengono?
# TODO si dovrebbe mettere una forte penalizzazione per gli outliers come una regolarizzazione L2?



# plot training history (accuracy and loss)

#matplotlib.rc('font', family='serif') 
#matplotlib.rc('font', serif='serif') 
#matplotlib.rc('text', usetex='false') 

#matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['font.serif'] = 'Palatino'
#font.serif          : DejaVu Serif, Bitstream Vera Serif, New Century Schoolbook, Century Schoolbook L, Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L, Times New Roman, Times, Palatino, Charter, serif

## list all data in history
#print(history.history.keys())
start_acc = 0.5 # accuracy nel caso di random guess a maximun entropy
start_loss = numpy.log(2) # misurando in base 2 si dovrebbe invece partire da 1
start_values = {'acc':start_acc, 'loss':start_loss, 'val_acc':start_acc, 'val_loss':start_loss}
# 1 è il massimo valore di binary_crossentropy (P = 1/2)
# 0.5 è il peggior valore di accuracy (random guess con P = 1/2)
start_values = pandas.DataFrame([start_values])
train_history = pandas.read_csv('./images/training_history.csv')
train_history = pandas.concat([start_values, train_history], ignore_index=True)
# accuracy over training epochs
#fig = pyplot.figure()
#ax1 = fig.add_subplot(111)
fig, [ax1, ax3] = pyplot.subplots(2, sharex=True)
fig.suptitle('model performances', size=16)
#ax1 = pyplot.subplot(2, 1, 1)
ax1.plot(1 - train_history['acc'], label='train')
ax1.plot(1 - train_history['val_acc'], label='test')
#ax1.set_title('model accuracy')
# 1- accuracy = error # TODO è lo stesso errore definito come y-d?
# TODO se non fosse lo stesso, sarebbe più giusto chiamarlo 'classification error'?
# TODO oppure 'misclassification'?
ax1.set_ylabel('classification error') # r'$1-$accuracy' = 'error'
#ax1.set_xlabel('epoch')
ax1.legend(loc='best')
#for i in ax1.yaxis.get_ticklabels():
#	i.set_horizontalalignment('right')
#ax1.set_yticklabels([], ha='right')
ax1.set_yscale('log')
sigma_levels = [0.6826895, 0.9544997, 0.9973002, 0.9999366, 0.9999994]
sigma_levels = numpy.array(sigma_levels)
#pyplot.axhline(y=1-0.6826895, color='red', alpha=0.5)
#pyplot.axhline(y=1-0.9544997, color='red', alpha=0.5)
#pyplot.axhline(y=1-0.9973002, color='red', alpha=0.5)
#pyplot.axhline(y=1-0.9999366, color='red', alpha=0.5)
#pyplot.axhline(y=1-0.9999994, color='red', alpha=0.5)
ax1.hlines(1-sigma_levels, *ax1.get_xlim(), linestyles='dotted', alpha=0.5)#, color='red')
ax2 = ax1.twinx()
#ax2 = pyplot.twinx()
ax2.set_yscale('log')
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticks(1-sigma_levels)
ax2.set_yticklabels([r'$1 \sigma$', r'$2 \sigma$', r'$3 \sigma$', r'$4 \sigma$', r'$5 \sigma$'])#, ha='left')
ax2.minorticks_off()
# loss over training epochs
#pyplot.subplot(2, 1, 2)
#ax3 = fig.add_subplot(112)
ax3.plot(train_history['loss'], label='train')
ax3.plot(train_history['val_loss'], label='test')
#ax3.set_title('model loss')
ax3.set_ylabel('cost')
# TODO label sulle ordinate 'binary cross-entropy'
##binary_crossentropy(predictions, true)
##binary cross-entropy between predictions and targets.
## L(p,t) = − t log(p) − (1−t) log(1−p)
##This is the loss function of choice for binary classification problems and sigmoid output units.
# TODO mettere casella di testo con valori di batch_size, dataset_size e validation_percentage
ax3.set_xlabel('epoch')
ax3.legend(loc='best', frameon=False)
ax3.set_yscale('log')
#pyplot.show()
fig.savefig('./media/training_history.svg')
pyplot.close()


## precision and recall plots
#start_values = {'precision':0.5, 'recall':1, 'val_precision':0.5, 'val_recall':1}
#start_values = pandas.DataFrame([start_values])
#train_history = pandas.read_csv('train_history.csv')
#train_history = pandas.concat([start_values, train_history], ignore_index=True)
#fig, [ax1, ax2] = pyplot.subplots(2, sharex=True)
#fig.suptitle('model performances', size=16)
#ax1.plot(train_history['precision'], label='train')
#ax1.plot(train_history['val_precision'], label='test')
#ax1.set_ylabel('precision') # TODO mettere range da 0.5 a 1
#ax1.legend(loc='best', frameon=False)
##ax1.set_yscale('log')
#ax2.plot(train_history['recall'], label='train')
#ax2.plot(train_history['val_recall'], label='test')
#ax2.set_ylabel('recall') # TODO mettere range da 0 a 1 a passi da 0.2
#ax2.set_xlabel('epoch')
#ax2.legend(loc='best', frameon=False)
##ax2.set_yscale('log')
##pyplot.show()
#fig.savefig('precision_recall.svg')
#pyplot.close()




threshold = 0.5

# plot an histogram of the classifier's predictions
# to gain a feeling of:
# - thresholding error
# - prediction confidence
# - class separability
fig_predictions = pyplot.figure()
ax1 = fig_predictions.add_subplot(111)

# predictions for true noise
n, bins, rectangles = ax1.hist(predicted_classes[true_classes == 0], 
	bins=50,
	range=(0,1),
	normed=True, 
	histtype='step', 
	#alpha=0.6,
	color='#ff3300',
	label='noise')
# predictions for true signal
n, bins, rectangles = ax1.hist(predicted_classes[true_classes == 1], 
	bins=50,
	range=(0,1),
	normed=True, 
	histtype='step', 
	#alpha=0.6,
	color='#0099ff',
	label='noise + signal')
ax1.set_title('classifier output') # OR 'model output'
ax1.set_ylabel('density')
ax1.set_xlabel('class prediction')
tick_spacing = 0.1
#ax1.set_yscale('log')
ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacing))
#ax1.legend(loc='best')
pyplot.axvline(x=threshold, 
	color='grey', 
	linestyle='dotted', 
	alpha=0.8)
ax1.legend(loc='upper left', frameon=False)
fig_predictions.savefig('./media/class_predictions.svg')
pyplot.close()




# compute confusion matrix

# quantize predictions (threshold = 0.5)
rounded_predicted_classes = numpy.rint(predicted_classes)
# a = model.predict_classes(validation_images)
# b = a.reshape(len(validation_images))
# c = rounded_predicted_classes.astype('int32').reshape(len(validation_images))
# # b e c sono la stessa identica cosa

binary_confusion_matrix = confusion_matrix(true_classes, rounded_predicted_classes, labels=None, sample_weight=None)

# save the matrix to disk
numpy.savetxt("./models/confusion_matrix.csv", binary_confusion_matrix, 
	delimiter=",", 
	fmt='%1u') # unsigned integers



# compute precision and recall

# 'p' for 'predicted', 't' for 'true'
[[p0t0,p1t0],[p0t1,p1t1]] = binary_confusion_matrix

precision = (p1t1)/(p1t1 + p1t0) # purity
recall = (p1t1)/(p1t1 + p0t1) # efficiency

numpy.savetxt("./models/model_performances.txt", [precision, recall], 
	delimiter=",", 
	header='purity (precision) and efficiency (recall)', 
	fmt='%1.6f') # 6-digit float

# TODO il mio calcolo non coincide con quello di keras alla terza cifra decimale
# è per caso colpa del K.epsilon a denominatore?





# come interpretare la confusion matrix:
# [1,1,1,1,1,0,0,0,0,0] # true
# [0,1,0,1,0,1,0,0,0,0] # predicted
# >>> array([[4, 1], [3, 2]])
# 4 veri zeri predetti giusti
# 2 veri uni predetti giusti
# 1 vero zero predetto sbagliato
# 3 veri uni predetti sbagliati
# dunque:
#           predicted 0   predicted 1
#         -----------------------------
# real 0  |      4      |      1      |
#         |---------------------------|
# real 1  |      3      |      2      |
#         -----------------------------
#


# TODO fare analisi al contrario e risalire all'unica immagine che è stata classificata non correttamente, per cercare di capire quale può essere stato il problema
# TODO compute (score, recall, precision) and bias-variance tradeoff
# TODO calcorare purezza ed efficienza del trigger
# TODO fare k-fold con k=2, invertendo i dataset di train e test
# efficiency (quanti eventi generati sono rivelati), background rejection, efficiency uncertainty, sensibility, ROC curve
# efficiency vs background rejection (tradeoff)
# purity = numero di eventi del tipo desiderato selezionati dal trigger / numero di eventi totali selezionati dal trigger = (p1t1)/(p1t0 + p1t1) = misura quanto il fascio selezionato è composto da ciò che realmente si vuole
# efficiency = numero di eventi selezionati del tipo giusto / numero totale di eventi prodotti del tipo giusto = numero eventi triggerati / numero totale eventi prodotti = (p1t1)/(p1t1 + p0t1) = misura quanti eventi desiderati sono riuscito a catturare rispetto a quelli prodotti
# accuracy = frazione di previsioni corrette = 1 - classification error = probability of a correct classification = (predicted = true)/(sample_lenght) = (p1t1 + p0t0)/(p1t1 + p1t0 + p0t1 + p0t0)
# precision is the ability of the classifier not to label as positive a sample that is negative = tp/(tp+fp) = (p1t1)/(p1t1 + p1t0)
# recall is the ability of the classifier to find all the positive samples = tp/(tp+fn) = (p1t1)/(p1t1 + p0t1) = sensitivity = the ability of the test to detect disease in a population of diseased individuals
# specificity = TN/(TN + FP) = the ability of the test to correctly rule out the disease in a disease-free population
# mi pare dunque di capire che precision = purity e che recall = efficiency = sensitivity
# precision = true_positives / predicted_positives
# recall = true_positives / possible_positives = tp/(fp+tp)
# plottare precision_recall_curve
# (classification?) error = 1 - accuracy
#
# purity = precision = events correctly classified as signal / all events classified as signal
# efficiency = recall = events correctly classified as signal / all available signal events


exit()




plt.subplots_adjust(hspace=0.5)


f = plt.figure()
ax = f.add_subplot(111)
ax.yaxis.tick_right()
#ax.yaxis.set_label_position("right")
plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.plot([2,3,4,5])
ax.set_xlabel("$x$ /mm")
ax.set_ylabel("$y$ /mm")
plt.show()

f, axarr = plt.subplots(2, sharex=True)

ax1 = plt.plot()
t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
plt.plot(t,s1,'b-')
plt.xlabel('t (s)')
plt.ylabel('exp',color='b')

ax2 = plt.twinx()
s2 = np.sin(2*np.pi*t)
ax2.plot(t, s2, 'r.')
plt.ylabel('sin', color='r')
plt.show()

plt.savefig('plot.svg')

plt.yticks(1-sigma_levels, (r'$1 \sigma$', r'$2 \sigma$', r'$3 \sigma$', r'$4 \sigma$', r'$5 \sigma$'))

plt.subplot2grid((3,2), (0,1), rowspan=3)

#http://vitadigitale.corriere.it/2016/12/27/intelligenza-artificiale-apple-segreti/

# TODO mettere il loss in scala logaritmica e salvare entrambi i grafici su disco
# TODO unificare l'asse x dei due subplot

# TODO unire tutto in un unico grafico, con due scale diverse a destra e a sinistra per accuracy e cost

# TODO mettere linee orizzontali nel grafico logaritmico della accuracy ai vari livelli di sigma

# TODO perché ho il loss_test sempre minore del loss_train? underfitting?

#From the plot of accuracy we can see that the model could probably be trained a little more as the trend for accuracy on both datasets is still rising for the last few epochs. We can also see that the model has not yet over-learned the training dataset, showing comparable skill on both datasets.

#From the plot of loss, we can see that the model has comparable performance on both train and validation datasets (labeled test). If these parallel plots start to depart consistently, it might be a sign to stop training at an earlier epoch.

