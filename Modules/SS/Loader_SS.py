#SEMSEGTUT HANDLING
from __future__ import print_function

import numpy as np
import tensorflow as tf

import ROOT as R
import os,time,sys

from larcv import larcv
from larcv.dataloader2 import larcv_threadio

from NEUnet import config_NEUnet as M
from Modules.SEMSEGTUT import config_SEMSEGTUT as C
import tempfile

larcv.load_pyutil()

#Type of problem being solved and way data is stored and name
variety 	= C.DATA['Variety']
process		= C.DATA['Process']
name		= M.MAIN['Data']
#Image Dimensions and Colours and Classes and Axes
hsize 		= C.DATA['H']
vsize 		= C.DATA['V']
colours 	= C.DATA['Colours']
cnumb 		= C.DATA['Classes']
#Size of the dataset
trainsize 	= C.DATA['Training Set Size']
testsize 	= C.DATA['Testing Set Size']
#Batchsize designates number of points in each separate chunk
#Stepnumber designates number batches to go over
trbatchsize 	= M.TRAINING['Training Batch Size']
tebatchsize	= M.TRAINING['Testing Batch Size']
#Files and Paths
trf 		= C.DATA['Training File']
tef 		= C.DATA['Testing File']
path 		= C.DATA['Data Path']
trpath 		= path + trf
tepath 		= path + tef
#Collect the Config File
#Train_cfg 	= C.DATA['Training CFG']
#Test_cfg 	= C.DATA['Testing CFG'] 
#Net_cfg		= C.DATA['Net CFG']  
     
##################################################################
#Set up the configs for activation later

#Will need to separate out the CFGS later on but ftm its fine
Train_cfg = \
"""
MainIO: {
  Verbosity:    3
  EnableFilter: false
  RandomAccess: 2
  RandomSeed:   123
  InputFiles:   ["/user/jhenzerling/work/NEUsoft/Modules/SS/Data/SSTrain2.root"]
  ProcessType:  ["BatchFillerImage2D","BatchFillerImage2D"]
  ProcessName:  ["main_data","main_label"]
  NumThreads: 4
  NumBatchStorage: 4

  ProcessList: {
    main_data: {
      Verbosity: 3
      ImageProducer: "sbndwire"
      Channels: [0]
    }
    main_label: {
      Verbosity: 3
      ImageProducer: "sbnd_cosmicseg"
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
  InputFiles:   ["/user/jhenzerling/work/NEUsoft/Modules/SS/Data/SSTest2.root"]
  ProcessType:  ["BatchFillerImage2D","BatchFillerImage2D"]
  ProcessName:  ["test_data","test_label"]
  NumThreads: 4
  NumBatchStorage: 4

  ProcessList: {
    test_data: {
      Verbosity: 3
      ImageProducer: "sbndwire"
      Channels: [0]
    }
    test_label: {
      Verbosity: 3
      ImageProducer: "sbnd_cosmicseg"
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
LOAD_FILE          '/user/jhenzerling/work/NEUsoft/Output/Graphs/ssnet_checkpoint/uresnet-8000'
AVOID_LOAD_PARAMS  []
LEARNING_RATE 	   1e-4
ITERATIONS         8000
MINIBATCH_SIZE     2
NUM_MINIBATCHES    16
DEBUG              False
TRAIN              False
TF_RANDOM_SEED     123
USE_WEIGHTS        False
REPORT_STEPS       1
SUMMARY_STEPS      1
CHECKPOINT_STEPS   1000
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
#FAILS HERE img->img_t0
t.initialize()

#Return the initialized network object through data class
result = t






