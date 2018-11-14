#MNIST Data Module
#Configuration Files
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import ROOT as ROOT
import os,time,sys
import tempfile
larcv.load_pyutil()


from NEUview import config_NEUview as NV
from Modules.SEMSEGTUT import config_SEMSEGTUT as C

number 		= NV.MAIN['Number']
classn		= NV.MAIN['Class']
index		= NV.MAIN['Index']
searchtype	= NV.MAIN['SearchType']
imtype		= NV.MAIN['ImageType']
channel		= NV.MAIN['ImageChannel']
path1		= NV.PATH['Path']
trf 		= C.DATA['Training File']
tef 		= C.DATA['Testing File']
path 		= '/user/jhenzerling/work/NEUsoft/Modules/SEMSEGTUT/Data/'
trpath 		= path + trf
tepath 		= path + tef
#Collect the Config File
Train_cfg 	= C.DATA['Training CFG']
Test_cfg 	= C.DATA['Testing CFG']   

#########################################################################
#num = number of loaded images

Train_cfg = \
"""
MainIO: {
  Verbosity:    3
  EnableFilter: false
  RandomAccess: 2
  RandomSeed:   123
  InputFiles:   ["/user/jhenzerling/work/NEUsoft/Modules/SEMSEGTUT/Data/train_15k.root"]
  ProcessType:  ["BatchFillerImage2D","BatchFillerImage2D"]
  ProcessName:  ["main_data","main_label"]
  NumThreads: 4
  NumBatchStorage: 4

  ProcessList: {
    main_data: {
      Verbosity: 3
      ImageProducer: "data"
      Channels: [0]
    }
    main_label: {
      Verbosity: 3
      ImageProducer: "segment"
      Channels: [0]
    }
  }
}      
"""
train_io_config = tempfile.NamedTemporaryFile('w')
train_io_config.write(Train_cfg)
train_io_config.flush()

Test_cfg = \
"""
TestIO: {
  Verbosity:    3
  EnableFilter: false
  RandomAccess: 2
  RandomSeed:   123
  InputFiles:   ["/user/jhenzerling/work/NEUsoft/Modules/SEMSEGTUT/Data/test_10k.root"]
  ProcessType:  ["BatchFillerImage2D","BatchFillerImage2D"]
  ProcessName:  ["test_data","test_label"]
  NumThreads: 4
  NumBatchStorage: 4

  ProcessList: {
    test_data: {
      Verbosity: 3
      ImageProducer: "data"
      Channels: [0]
    }
    test_label: {
      Verbosity: 3
      ImageProducer: "segment"
      Channels: [0]
    }
  }
}      
"""

test_io_config = tempfile.NamedTemporaryFile('w')
test_io_config.write(Test_cfg)
test_io_config.flush()

net_config = \
"""
NUM_CLASS          3
BASE_NUM_FILTERS   16
MAIN_INPUT_CONFIG  '{:s}'
TEST_INPUT_CONFIG  '{:s}'
LOGDIR             '/user/jhenzerling/work/NEUsoft/Output/Graphs/ssnet_train_log'
SAVE_FILE          '/user/jhenzerling/work/NEUsoft/Output/Graphs/ssnet_checkpoint/uresnet'
LOAD_FILE          ''
AVOID_LOAD_PARAMS  []
ITERATIONS         8000
MINIBATCH_SIZE     20
NUM_MINIBATCHES    1
DEBUG              False
TRAIN              True
TF_RANDOM_SEED     123
USE_WEIGHTS        False
REPORT_STEPS       200
SUMMARY_STEPS      20
CHECKPOINT_STEPS   100
CHECKPOINT_NMAX    20
CHECKPOINT_NHOUR   0.4
KEYWORD_DATA       'main_data'
KEYWORD_LABEL      'main_label'
KEYWORD_WEIGHT     ''
KEYWORD_TEST_DATA  'test_data'
KEYWORD_TEST_LABEL 'test_label'
KEYWORD_TEST_WEIGHT ''
"""
ssnet_config = tempfile.NamedTemporaryFile('w')
ssnet_config.write(net_config.format(train_io_config.name, test_io_config.name))
ssnet_config.flush()

NetPath = '/user/jhenzerling/work/NEUsoft/Networks/u-resnet/lib'
sys.path.insert(0,NetPath)
import ssnet_trainval as api
t = api.ssnet_trainval()
t.override_config(ssnet_config.name)
t.initialize()

#######################################################################

def get_entry(entry):
    # image
    chain_image2d = ROOT.TChain("image2d_data_tree")
    chain_image2d.AddFile(tepath)
    chain_image2d.GetEntry(entry)
    cpp_image2d = chain_image2d.image2d_data_branch.as_vector().front()
    # label
    chain_label2d = ROOT.TChain("image2d_segment_tree")
    chain_label2d.AddFile(tepath)
    chain_label2d.GetEntry(entry)
    cpp_label2d = chain_label2d.image2d_segment_branch.as_vector().front()    
    return (np.array(larcv.as_ndarray(cpp_image2d)), np.array(larcv.as_ndarray(cpp_label2d)))

dir = path1 + '/Output/Images/SEMSEGTUT'
uv = []
uc = []
def showImage():
	for x in range(number):
		image2d, label2d = get_entry(x)
		uvi, uci = np.unique(label2d, return_counts=True)
		uv.append(uvi)
		uc.append(uci)
    		#print('Label values:',unique_values)
   		#print('Label counts:',unique_counts)
		fig, (ax0,ax1) = plt.subplots(1,2,figsize=(16,8), facecolor='w')
		ax0.imshow(image2d, interpolation='none', cmap='jet', vmin=0, vmax=1000, origin='lower')
		ax0.set_title('image',fontsize=24)
		ax1.imshow(label2d, interpolation='none', cmap='jet', vmin=0, vmax=3.1, origin='lower')
		ax1.set_title('label',fontsize=24)
		namer = 'SEMSEGTUT-' + str(x)
		plt.savefig(dir+"/"+namer)
		plt.show()
		plt.close()	
def showInfo():
	for x in range(number):
		print('Image Index %s has Labels %s in amount %s' % (x,uv[x],uc[x]))












    
