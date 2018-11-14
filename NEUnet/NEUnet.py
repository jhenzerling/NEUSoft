#NEUnet
#################################################################################################################
#LIBRARIES AND BACKGROUND
#File Handling
import sys,os
#Suppress extra warnings for readability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#Machine Learning Handling
from Objects import Loader as L
from Objects import Networker as N
from Objects import Trainer as T
from Objects import Finder as F
#Tensorflow Libraries
import tensorflow as tf
#Main Configuration
import config_NEUnet as CONFIG
#Reset
tf.reset_default_graph()
#################################################################################################################
#SETTINGS AND EDITING
#Network ON/OFF - Useful for Editing
trigger 	= CONFIG.MAIN['Trigger']
#Data Selection
datatrigger 	= CONFIG.MAIN['Data']
#If want to use TensorBoard
graphtrigger 	= CONFIG.MAIN['Graph']
#################################################################################################################
def NEUnet():
	print('Initial Learning Rate: ', CONFIG.TRAINING['Learn Init'])
	#Load the Data and Pather
	pather = F.Finder()
	data = L.Loader(datatrigger,pather)
	#Set up the Network
	if trigger == True:
		NW = N.Network(data)
   		if graphtrigger == True:
    		    	print('Producing TensorBoard Graph.')
		#Add prints here to display config info
		#Perform the Training
   	 	Learn = T.Trainer(data,NW)
   		Learn.RunTraining()
	else:
    		print('Trigger not ON. Terminating')
    		quit()
###########################################################################################################
