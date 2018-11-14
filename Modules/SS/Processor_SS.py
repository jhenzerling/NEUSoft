#PMULTI Preprocessor - Create and Store the Label Files in Background for storing later
#Configuration Files
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import ROOT as R
DB = R.TDatabasePDG()
import os,time,sys
import tempfile
larcv.load_pyutil()

import h5py


from NEUview import config_NEUview as NV
from Modules.SS import config_SS as C

number 		= NV.MAIN['Number']
classn		= NV.MAIN['Class']
index		= NV.MAIN['Index']
searchtype	= NV.MAIN['SearchType']
imtype		= NV.MAIN['ImageType']
channel		= NV.MAIN['ImageChannel']
path1		= NV.PATH['Path']
path 		= '/user/jhenzerling/work/NEUsoft/Modules/SS/Data/'

dname = 'SSData.root'
fpath = path + dname

entry = 0

##################################

def CosmicSegger(a):
	proc = larcv.ProcessDriver('ProcessDriver')
	#ABSOLUTELY NEED DOUBLE QUOTES??
	if a == 1:
		proc.configure('/user/jhenzerling/work/NEUsoft/Modules/SS/NEWCFG.cfg')
	elif a == 2:
		proc.configure('/user/jhenzerling/work/NEUsoft/Modules/SS/NEWCFG2.cfg')
	else:
		print('why')
	proc.initialize()
	proc.batch_process()
	proc.finalize()

def showCosSeg(which):
	deg = 8
	counter = np.zeros(deg)
	counter2 = np.zeros(deg)
	d1 = R.TChain('image2d_sbndwire_tree')
	d2 = R.TChain('image2d_sbnd_cosmicseg_tree')
	if which == 1:
		dname3 = 'SSTrain2.root'
		amount = 9000
	elif which == 2:
		dname3 = 'SSTest2.root'
		amount = 8500
	else:
		print('eff')
	fpath3 = '/user/jhenzerling/work/NEUsoft/Modules/SS/Data/' + dname3
	d1.AddFile(fpath3)
	d2.AddFile(fpath3)
	print(dname3)
	for entry in range(amount):
		if entry%500 == 0:
			print("entry= ", entry)
		d1.GetEntry(entry)
		d2.GetEntry(entry)
		d1b = d1.image2d_sbndwire_branch
		d2b = d2.image2d_sbnd_cosmicseg_branch
		d1v = d1b.as_vector()
		d2v = d2b.as_vector()
		d1i = larcv.as_ndarray(d1v.front())
		d2i = larcv.as_ndarray(d2v.front())
		unique_values, unique_counts = np.unique(d2i, return_counts=True)
		for i in range(deg):
			counter[i] += np.count_nonzero(d2i == i)
			if i in unique_values:
				counter2[i] += 1
		#plt.imshow(d2i,cmap=plt.get_cmap())
		#if entry == 0:
		#	cbar = plt.colorbar(ticks = [0,1,2,3,4,5,6,7])
		#	cbar.set_ticklabels(['0=Background','1=Photon','2=Electron','3=Muon','4=Pi0','5=PiC','6=Proton','7=Other'])
		#plt.savefig(path1 + '/Output/Images/SS/event_%s_GEN.png' % (entry),dpi=1000)
	#plt.show()
	print('partcount all, ', counter)
	print('partcount5, ', counter2)

def showCosSeg2(ted,which,entry):
	deg = 7
	counter = np.zeros(deg)
	d1 = R.TChain('image2d_sbndwire_tree')
	d2 = R.TChain('image2d_sbnd_cosmicseg_tree')
	if which == 1:
		dname3 = 'SSTrain.root'
		amount = 9000
	elif which == 2:
		dname3 = 'SSTest.root'
		amount = 8500
	else:
		print('eff')
	print(dname3)
	fpath3 = '/user/jhenzerling/work/NEUsoft/Modules/SS/Data/' + dname3
	d1.AddFile(fpath3)
	d2.AddFile(fpath3)
	d1.GetEntry(entry)
	d2.GetEntry(entry)
	d1b = d1.image2d_sbndwire_branch
	d2b = d2.image2d_sbnd_cosmicseg_branch
	d1v = d1b.as_vector()
	d2v = d2b.as_vector()
	d1i = larcv.as_ndarray(d1v.front())
	d2i = larcv.as_ndarray(d2v.front())
	if ted == 1:
		plt.imshow(d1i,cmap=plt.get_cmap())
	elif ted == 2:
		plt.imshow(d2i,cmap=plt.get_cmap())
	else:
		print('bruh')
	cbar = plt.colorbar(ticks = [0,1,2,3,4,5,6])
	cbar.set_ticklabels(['0=Background','1=Elec','2=Muon','3=Phot','4=Proton','5=Other(K/PiC)'])
	#plt.savefig(path1 + '/Output/Images/SS2/event_%s_GEN.png' % (entry),dpi=1000)
	plt.show()


def checkpdg():

	path1 = '/hepstore/jhenzerling/sbnd_dl_samples/sbnd_dl_cosmics_larcv_test.root'
	c1 = R.TChain('particle_sbndseg_tree') #particle info
	c1.AddFile(path1)
	amount = c1.GetEntries()
	store = []
	print(amount)
	#c1.GetEntry(1)
	#branchp = c1.particle_sbndseg_branch
	#print(dir(branchp.as_vector()))
	#print(branchp.as_vector().size())
	for y in range(amount):	
		if y%500 == 0:
			print('entry = ', y)
		c1.GetEntry(y)
		branchp = c1.particle_sbndseg_branch
		for x in range(branchp.as_vector().size()):
			pv = abs(branchp.as_vector()[x].pdg_code())
			if pv not in store:
				store.append(pv)
				print('added new pdg: ', pv)
	print(store)





#8.5k TRAIN,9k TEST
#CosmicSegger(1)
#CosmicSegger(2)
#checkpdg()
#showCosSeg(1)
#showCosSeg(2)
showCosSeg2(2,1,2069)
#CosmicSegger(1)

def Process(a,b):
	return 0
