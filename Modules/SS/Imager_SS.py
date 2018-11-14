#Imager for PMULTI Testing
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


from NEUview import config_NEUview as NV
from Modules.PMULTI import config_PMULTI as C

number 		= NV.MAIN['Number']
classn		= NV.MAIN['Class']
index		= NV.MAIN['Index']
searchtype	= NV.MAIN['SearchType']
imtype		= NV.MAIN['ImageType']
channel		= NV.MAIN['ImageChannel']
path1		= NV.PATH['Path']
#trf 		= C.DATA['Training File']
#tef 		= C.DATA['Testing File']
path 		= '/user/jhenzerling/work/NEUsoft/Modules/PMULTI/Data/'
#trpath 		= path + trf
#tepath 		= path + tef
#Collect the Config File
#Train_cfg 	= C.DATA['Training CFG']
#Test_cfg 	= C.DATA['Testing CFG']   

dname = 'sbnd_dl_cosmics_larcv_dev.root'
fpath = path + dname

entry = 0

##################################
#The Branches
c1 = R.TChain('particle_sbndseg_tree') #particle info
c2 = R.TChain('cluster2d_sbndseg_tree') #image info
c3 = R.TChain('image2d_sbndwire_tree') #seg info
#c4 = R.Tchain('cluster3d_sbndseg_tree')

c1.AddFile(fpath)
c2.AddFile(fpath)
c3.AddFile(fpath)
#c4.AddFile(fpath)

c1.GetEntry(entry)
c2.GetEntry(entry)
c3.GetEntry(entry)
#c4.GetEntry(entry)

c1b = c1.particle_sbndseg_branch
c2b = c2.cluster2d_sbndseg_branch
c3b = c3.image2d_sbndwire_branch
##################################

def displayImage(chain,trig,eentry):
	chain.GetEntry(eentry)
	cbv = chain.image2d_sbndwire_branch.as_vector()
	im2d = larcv.as_ndarray(cbv.front())

	plt.title('Event %d Image2D' % (eentry))
	plt.imshow(im2d,cmap=plt.get_cmap())
	plt.savefig(path1 + '/Output/Images/PMULTI/event_%s.png' % (eentry))
	if trig == True:
		plt.show()
		plt.close()
	
#displayImage(c3,True,0)
#THIS SHOWS THE RAW IMAGE! YES!!

##################################

#Convert from classnumb to pdg
def pdglist(i):
	if i == 0:
		pdg = 0
	elif i == 1:
		pdg = 11
	elif i == 2:
		pdg = 13
	elif i == 3:
		pdg = 22
	elif i == 4:
		pdg = 211
	elif i == 5:
		pdg = 2212
	elif i == 6:
		pdg = 9999
	else:
		quit()
	return pdg

def antipdglist(pdg):
	if abs(pdg) == 0:
		x = 0
	elif abs(pdg) == 11:
		x = 1
	elif abs(pdg) == 13:
		x = 2
	elif abs(pdg) == 22:
		x = 3
	elif abs(pdg) == 211:
		x = 4
	elif abs(pdg) == 2212:
		x = 5
	elif abs(pdg) == 9999:
		x = 6
	else:
		quit()
	return x	

def clus2Arr(branch,index):
	clus = branch.cluster_pixel_2d(0).as_vector()
	image2d = larcv.as_image2d(clus[index],branch.cluster_pixel_2d(0).meta())
	image2d = larcv.as_ndarray(image2d)*100
	return image2d

#Outputs the indexes where there are nonzero tracks
def findTracks(segb,pb):
	result = []
	pbv = pb.as_vector()
	for index,particle in enumerate(pbv):
		clus = segb.cluster_pixel_2d(0).as_vector()[index]
		if clus.as_vector().size() != 0:
			result.append(index)
	return result

def displayOneCluster(branch,index):
	im = clus2Arr(branch,index)
	plt.title('Cluster %d' % (index))
	plt.imshow(im,cmap=plt.get_cmap())
	plt.show()

#Displays One Cluster that's One-Hotted (Labelled)
#Need to combine these in order to produce a segment map
def cluster2PDG(branchp,branchs,index):
	im = clus2Arr(branchs,index)
	pv = branchp.as_vector()[index].pdg_code()
	LABEL = antipdglist(pv)
	nonzeros = np.where(im != 0)
	nonzstore = []
	for x in range(len(nonzeros[0])):
		nonzstore.append([nonzeros[0][x],nonzeros[1][x]])

	start = np.zeros([1280,1986],dtype=np.float32)
	end = start
	for y in range(len(nonzstore)):
		[xn,yn] = nonzstore[y]
		end[xn,yn] = LABEL
	return [end,nonzstore]

def pixelmatcher(n1,n2):
	res = []
	if len(n1) <= len(n2):
		for x in range(len(n1)):
			if n1[x] in n2:
				res.append(n1[x])
	else:
		for x in range(len(n2)):
			if n2[x] in n1:
				res.append(n2[x])
	return res		

def overlay(im1,im2,n1,n2):
	olay = im1 + im2
	matches = pixelmatcher(n1,n2)
	n3 = np.concatenate((n1,n2),axis=0)
	if matches == []:
		return [olay,n3]
	else:
		for x in range(len(matches)):
			L1 = im1[matches[x][0],matches[x][1]]
			L2 = im2[matches[x][0],matches[x][1]]
			if L1 < L2:
				olay[matches[x][0],matches[x][1]] = L2
			elif L2 < L1:
				olay[matches[x][0],matches[x][1]] = L1
			else:
				olay[matches[x][0],matches[x][1]] = L1
	return [olay,n3]

def displayLClusters(branchp,branchs):
	indexes = findTracks(branchs,branchp)
	for x in range(len(indexes)):
		if x == 0:
			pass
		elif x == 1:
			[im1,N1] = cluster2PDG(branchp,branchs,indexes[x])
			[im2,N2] = cluster2PDG(branchp,branchs,indexes[x-1])
			[im3,N3] = overlay(im1,im2,N1,N2)
		else:
			[im1,N1] = cluster2PDG(branchp,branchs,indexes[x])
			[im3,N3] = overlay(im1,im3,N1,N3)
			
	plt.title('Label')
	plt.imshow(im3,interpolation='none',vmin = 0, vmax=6, cmap='jet')
	plt.show()	




displayLClusters(c1b,c2b)



#Clusters are not as bright as image2d, there's some scaling factor I've yet to figure out
#The scaling factor seems very large even though I've added in the 100x in each cluster
def displayClusters(branchp,branchs):
	indexes = findTracks(branchs,branchp)
	for x in range(len(indexes)):
		if x == 0:
			im = clus2Arr(branchs,indexes[x])
		else:
			im += clus2Arr(branchs,indexes[x])
	plt.title('all clusters')
	plt.imshow(im,cmap=plt.get_cmap())
	plt.show()



###################################################


#Class Vector to PDG Vector
def class2PDG(enter):
	for x in range(len(enter)):
		if enter[x] == 1:
			result = pdglist(x)
	return result

################################################


def showImage():
	return 0
def showInfo():
	return 0

