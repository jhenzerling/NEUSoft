#CIFAR10 Data Module
#Configuration Files
import matplotlib.pyplot as plt
import os,pickle
import pandas as pd
import numpy as np
from NEUview import config_NEUview as NV
from Modules.MNIST import config_MNIST as C

number 		= NV.MAIN['Number']
classn		= NV.MAIN['Class']
index		= NV.MAIN['Index']
searchtype	= NV.MAIN['SearchType']
path		= NV.PATH['Path']
channel		= NV.MAIN['ImageChannel']

#########################################################################    
def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo)
	return dict
def get_data(file):
	absFile = path + '/Modules/' + NV.MAIN['Data'] + '/Data/' + file
	dict = unpickle(absFile)
	#for key in dict.keys():
	#	print(key)
	#print("Unpacking {}".format(dict[b'batch_label']))
	X = np.asarray(dict[b'data'].T).astype("uint8")
	Yraw = np.asarray(dict[b'labels'])
	Y = np.zeros((10,10000))
	for i in range(10000):
		Y[Yraw[i],i] = 1
	names = np.asarray(dict[b'filenames'])
	return X,Y,names
def visualize_image(X,Y,names,id):
	rgb = X[:,id]
	img = rgb.reshape(3,32,32).transpose([1, 2, 0])
	plt.imshow(img)
	plt.title(names[id])
	plt.show()
	dir = path + '/Output/Images/CIFAR10'
	plt.savefig(dir+"/"+names[id].decode('ascii'))

def searchClass(num, total, Y):
	counter = []
	for x in range(10000):
		if Y[num][x] == 1:
			if len(counter) < total:
				counter.append(x)
			else: 
				break
	return counter

def searchIndex(total):
	counter = []
	for x in range(total):
		counter.append(x)
	return counter

#########################################################################
	
spec = 'data_batch_' + str(channel)
X,Y,names = get_data(spec)
if searchtype == 'Index':
	pnumb = searchIndex(number)
elif searchtype == 'Class':
	pnumb = searchClass(classn,number, Y)
else:
	pnumb = []

#########################################################################

def showImage():
    for x in range(len(pnumb)):
	visualize_image(X,Y,names,pnumb[x])

def showInfo():
    for x in range(len(pnumb)):
	print('Image index %s has label %s' % (pnumb[x],Y[:,pnumb[x]]))

##################################################################

		



    
