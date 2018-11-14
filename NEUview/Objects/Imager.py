#Imager class for handling image display, will add Imager_DATA to Modules
import os,sys,os.path
import importlib as i
from NEUview import config_NEUview as CONFIG

class Imager:
	#Instantiate Paths
	def __init__(self):
		Mpath = CONFIG.PATH['Path']
		Ipath = Mpath + '/Modules/' + CONFIG.MAIN['Data'] + '/'
		Dpath = Ipath +  '/Data/'
		imagername = 'Imager_' + CONFIG.MAIN['Data']
		imagermodloc = 'Modules.' + CONFIG.MAIN['Data']
		Ifile = Ipath + imagername + '.py' 
		
		if(os.path.exists(Dpath)):
   			print(CONFIG.MAIN['Data'] + ' Data Present.')
			#Check for Imager
			if(os.path.isfile(Ifile)):
   				print(CONFIG.MAIN['Data'] + ' Imager Present.')
				try:
					self.Imod = i.import_module('.' + imagername, package = imagermodloc)
				except ImportError:
					sys.exit('Failed to import, imager name fail: ' + imagermodloc)	
			else:
    				print(CONFIG.MAIN['Data'] + ' Imager Not Present.')    		
				quit()
		else:
    			print(CONFIG.MAIN['Data'] + ' Data Not Present.')
    			quit()

	###################################

	def showImage(self):
		self.Imod.showImage()
	def showInfo(self):
		self.Imod.showInfo()
