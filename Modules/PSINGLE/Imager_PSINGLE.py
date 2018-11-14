#MNIST Data Module
#Configuration Files
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import ROOT as R
import time
larcv.load_pyutil()


from NEUview import config_NEUview as NV
from Modules.PSINGLE import config_PSINGLE as C

number 		= NV.MAIN['Number']
classn		= NV.MAIN['Class']
index		= NV.MAIN['Index']
searchtype	= NV.MAIN['SearchType']
imtype		= NV.MAIN['ImageType']
channel		= NV.MAIN['ImageChannel']
path1		= NV.PATH['Path']
trf 		= C.DATA['Training File']
tef 		= C.DATA['Testing File']
path 		= '/user/jhenzerling/work/NEUsoft/Modules/PSINGLE/Data/'
trpath 		= path + trf
tepath 		= path + tef
#Collect the Config File
Train_cfg 	= C.DATA['Training CFG']
Test_cfg 	= C.DATA['Testing CFG']   

#########################################################################
#num = number of loaded images

#Construct and prepare memory for the threadio
def IOPrep(name):
    if(name == 'Train'):
        cfg = Train_cfg
	b = 100
    elif(name == 'Test'):
        cfg = Test_cfg
	b = 100
    else:
        print('Bad name, check ImageType')
    	quit()

    proc = larcv_threadio()
    proc.configure(cfg)
    proc.start_manager(b)
    #Need sleep for manager to finish loading
    time.sleep(2)
    proc.next()

    return proc

#Convert the flat data into a 2D image for use in the network
def formatTensors(da,bs):
    image_tensor = tf.convert_to_tensor(da[0])
    image_tensor_2d = tf.reshape(image_tensor, [bs, hsize, vsize, colours])
    label_tensor = tf.convert_to_tensor(da[1])
    
    return [image_tensor_2d, label_tensor]


def searchIndex(total):
	counter = []
	for x in range(total):
		counter.append(x)
	return counter
def searchClass(num,labels):
	counter = []
	for x in range(100):
		if labels[x][classn] == 1:
			if len(counter) < num:
				counter.append(x)
	return counter

if imtype == 'Train':
	named = 'train_image'
	namel = 'train_label'
elif imtype == 'Test':
	named = 'test_image'
	namel = 'test_label'
else:
	print('check imtype')
	quit()

proc = IOPrep(imtype)
data = proc.fetch_data(named)
lata = proc.fetch_data(namel)

if searchtype == 'Index':
	pnumb = searchIndex(number)
elif searchtype == 'Class':
	pnumb = searchClass(number,lata.data())
else:
	pnumb = []
print(pnumb)

dir = path1 + '/Output/Images/PSINGLE'
def showImage():
	for x in range(len(pnumb)):
		imaged = data.data()
		imagedim = data.dim()
		image2d = imaged.reshape(imagedim[:-1])
		fig,ax = plt.subplots(figsize=(8,8))
		plt.imshow(image2d[pnumb[x]],cmap='jet',interpolation='none')
		plt.title(lata.data()[pnumb[x]])
		namer = 'PSINGLE-' + str(pnumb[x]) + '-'.format(lata.data()[pnumb[x]])
		plt.savefig(dir+"/"+namer)
		plt.show()
		plt.close()
		

def showInfo():
	print('For Channel 0: (If want change channel access the cfg)')
	for x in range(len(pnumb)):
		print('Image index %s has label %s' % (index,lata.data()[pnumb[x]]))

		



    
