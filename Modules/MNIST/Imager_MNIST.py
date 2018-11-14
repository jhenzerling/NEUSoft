#MNIST Data Module
#Configuration Files
import matplotlib.pyplot as plt
import numpy as np
from NEUview import config_NEUview as NV
from Modules.MNIST import config_MNIST as C

#Pull data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./Modules/MNIST/Data/', one_hot=True)

number 		= NV.MAIN['Number']
classn		= NV.MAIN['Class']
index		= NV.MAIN['Index']
searchtype	= NV.MAIN['SearchType']
path		= NV.PATH['Path']

#########################################################################
#num = number of loaded images
def load_images(num):
    x_train = mnist.train.images[:num,:]
    y_train = mnist.train.labels[:num,:]
    x_test = mnist.test.images[:num,:]
    y_test = mnist.test.labels[:num,:]
    return [x_train, y_train, x_test, y_test]
    
def searchClass(num, total):
	counter = []
	for x in range(55000):
		if y_train[x].argmax(axis=0) == num:
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
	
[x_train,y_train,x_test,y_test] = load_images(55000)

if searchtype == 'Index':
	pnumb = searchIndex(number)
elif searchtype == 'Class':
	pnumb = searchClass(classn,number)
else:
	pnumb = []	

#########################################################################

def showImage():
    for x in range(len(pnumb)):
    	label = y_train[pnumb[x]].argmax(axis=0)
    	image = x_train[pnumb[x]].reshape([28,28])
    	plt.title('Example: %d  Label: %d' % (pnumb[x], label))
    	plt.imshow(image, cmap=plt.get_cmap())
	plt.savefig(path + '/Output/Images/MNIST/index-%s-label-%s.png' % (pnumb[x],label))
    	plt.show()	

def showInfo():
    for x in range(len(pnumb)):
	label = y_train[pnumb[x]].argmax(axis=0)
    	print('Image index %s has label %s' % (pnumb[x],label))

##################################################################

		



    
