import sys,timeit
start = timeit.default_timer()
from NEUnet import NEUnet as NN
from NEUview import NEUview as NV
from NEUprocessor import NEUprocessor as NP
################################################################################################
NN.NEUnet()

#NP.NEUprocessor()

#NV.NEUview()
################################################################################################

stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)
sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))
