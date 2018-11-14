#NEU preprocessing package
#################################################################################################################
#LIBRARIES AND BACKGROUND
#File Handling
import sys,os
#Suppress extra warnings for readability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from Objects import Processor as P
#Main Configuration
import config_NEUprocessor as CONFIG
#################################################################################################################
#CONFIG PASSING
outputpath = CONFIG.PATH['Path'] + '/Modules/' + CONFIG.MAIN['Data'] + '/Data/'
options = CONFIG.MAIN['Options']
#################################################################################################################
def NEUprocessor():
	print('Performing PreProcessing for ', CONFIG.MAIN['Data'])
	PR = P.Processor()
	PR.preProcess(outputpath,options)
	return 0

#####################################################################################

