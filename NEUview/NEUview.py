#NEUnet
#################################################################################################################
#LIBRARIES AND BACKGROUND
#File Handling
import sys,os
#Suppress extra warnings for readability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from Objects import Imager as I
#Main Configuration
import config_NEUview as CONFIG
#################################################################################################################
#CONFIG PASSING

#################################################################################################################
def NEUview():
	showNVinfo()
	#Call the search/display funct
	IM = I.Imager()
	
	#Display Info of Images
	IM.showImage()
	IM.showInfo()

	return 0
###########################################################################################################

def showNVinfo():
	if CONFIG.MAIN['Number'] > 1:
		print('From dataset %s, displaying %s images of type %s, searching by %s' % (CONFIG.MAIN['Data'],CONFIG.MAIN['Number'],CONFIG.MAIN['ImageType'],CONFIG.MAIN['SearchType']))
	elif CONFIG.MAIN['Number'] == 1:
		print('From dataset %s, displaying %s image of type %s, searching by %s' % (CONFIG.MAIN['Data'],CONFIG.MAIN['Number'],CONFIG.MAIN['ImageType'],CONFIG.MAIN['SearchType']))
	else:
		print('Check image display number')
		exit()
