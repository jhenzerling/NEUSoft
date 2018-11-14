#Configuration for PSINGLE Data
#Configuration for Data
DATA= {
    	'H'       		: 256,
    	'V'       		: 256,
    	'Colours' 		: 1,
	'Axes'			: 1,
	'Classes' 		: 5,
    	'Training Set Size'	: 50000,
	'Testing Set Size'	: 40000,
	'Training File'     	: 'Data/train_50k.root',
   	'Training CFG'   	: {'filler_name':'TrainCFG','verbosity':0,'filler_cfg':'Modules/PSINGLE/TrainCFG.cfg'},
    	'Testing File'     	: 'Data/test_40k.root',
    	'Testing CFG'   	: {'filler_name':'TestCFG','verbosity':0,'filler_cfg':'Modules/PSINGLE/TestCFG.cfg'},
	'Variety'		: 'Classification',
	'Process'		: 'Proc'	
}

