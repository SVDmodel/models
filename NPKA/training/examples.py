# -*- coding: utf-8 -*-
"""
############################################################################################
 DNN Training utilties

Rammer, W, Seidl, R: A scalable model of vegetation transitions using deep learning
#############################################################################################
"""

## create a thread-safe generator


import pandas as pd
import numpy as np
from queue import Queue, Full
from threading import Thread, current_thread
from datetime import datetime

path_to_data=""

def loadExamplesRun(run_id):
    global path_to_data
    log("loading data for run " + str(run_id) + ", " + str(current_thread()))
    examples = pd.read_csv(path_to_data + "/all.pruned_"+str(run_id) + ".csv")
    examples = examples.values
    ## to shuffle randomy: 
    ridx = np.arange(len(examples), dtype=np.int32)
    np.random.shuffle(ridx) # shuffle in place
    log("loading finished: " + str(current_thread()))
    return examples, ridx  



def buildExamples2(batchsize, bi, env, examples, ridx):  
    # for now, we ignore the very last examples
    idx = ridx[bi*batchsize:(bi+1)*batchsize]
    row = examples[idx]
    # 0: ustateid, 1: nextState, 2: year, 3: prevYear, 4: nextYear, 5: rid, 6: runid, 7: l_piab, ... 68: m_rops
    
    #cdat = np.array([env.climateData(cellId=row['rid'][r], year=int(row['year'][r]), runId=int(row['runId'][r])) for r in idx], dtype=np.float32)
    cdat = np.array([env.climateData(cellId=int(row[r, 5]), year=int(row[r, 2]), runId=int(row[r, 6])) for r in range(len(row))], dtype=np.float32)
    
    sitedata = np.array( [env.soilData(cellId=row[r, 5]) for r in range(len(row)) ])
    labs = np.array(row[:, 1], dtype=np.int16) - 1 # input states: 1..N, we want: 0..N-1
    labs_time = np.array(row[:, 4], dtype=np.int16)-1
    restime = np.array(row[:, 3]/10, dtype=np.int16) # the time *already* in the state, in decades
    state = np.array(row[:, 0], dtype=np.int16) - 1 ## to 0-based
    neighbors = np.array(row[:, 7:69], dtype=np.float16)
    distance = np.array(row[:, 69], dtype=np.float16) 
    return( {'state_input': state,
            'site_input': sitedata,
            'time_input': restime,
            'neighbor_input': neighbors,
            'clim_input': cdat,
            'distance_input': distance},
            { 'out': labs,
             'time_out': labs_time })




# the queues:
test_queue = Queue(maxsize=100)
val_queue = Queue(maxsize=10)

#pool_sema = BoundedSemaphore(value=10+10)

run_threads = True
run_threads = False

env = None

# worker function....
def getExample(q, validation, run_id_list, batch_size):
    run_list_index = 0
    max_batches = -1
    i = 0
    while run_threads:
        if i > max_batches:
           examples, ridx = loadExamplesRun(run_id_list[run_list_index])
           max_batches = int(len(examples)/batch_size)-1
           log("switched data. new batches: " + str(max_batches+1))
           i=0
           run_list_index = (run_list_index + 1 ) % len(run_id_list)
           
        batch = buildExamples2(batch_size, i, env, examples, ridx)
        i = i + 1
        #if i%1000 == 0:
        #    print(str(current_thread()) + ": " + str(i) + ", val: " + str(val_queue.qsize()) + ", test: " + str(test_queue.qsize()) )
        
        if validation:
            q=val_queue
        else:
            q=test_queue
            
        stored = False
        while not stored:
            try:
                q.put(batch, block=True, timeout=1)
                stored=True
                #print("stored")
            except Full:
                # check for cancel
                if not run_threads:
                    break
                

    log("thread finished: " + str(current_thread()))
        
lgfile=None 
def startLogging(log_file):
    global lgfile
    lgfile = open(log_file, "w")
    log("Logging started.")

def stopLogging():
    global lgfile
    log("Stopping logging")
    lgfile.close()
    
def log(s):
    global lgfile
    lgfile.write(str(datetime.now()) + ": " + s + "\n")
    lgfile.flush()
    

### Start threads
workers = []
def startWorkers(batch_size, environment, train_files, val_files, threads_train=3, threads_val=2, log_file='log.txt', datapath=''):
    global run_threads, env
    global path_to_data
    path_to_data = datapath
    startLogging(log_file)
    env = environment 
    run_threads = True
    #runs = [[1,2], [31,32], [11,12], [21,22]  ]
    runs_train = np.array_split(train_files, threads_train)
    runs_val = np.array_split(val_files, threads_val)
    # orig: runs = [[1,2], [31,32], [11,12], [21,22]  ]
    for i in range(threads_train+threads_val):
        #runs = [e+10*i for e in [1,2]]
        if i<threads_train:
            worker = Thread(target=getExample, args=(test_queue,False, runs_train[i], batch_size)) # all test files
        else:
            worker = Thread(target=getExample, args=(val_queue,True, runs_val[i - threads_train], batch_size)) ## validation files
            
        worker.setDaemon(True)
        worker.start()
        workers.append( worker )
  

def stopWorkers():
    global workers, run_threads
    run_threads = False

    for worker in workers:
        worker.join()
        
    print("all threads stopped.")
    workers = []
    stopLogging()
  
    
def queueState():
    print("val: " + str(val_queue.qsize()) + ", test: " + str(test_queue.qsize()) )    

def fetchExamples(val):
    while True:
        if val:
            yield val_queue.get(timeout=100)
            #pool_sema.release()
            val_queue.task_done()
        else:
            yield test_queue.get(timeout=100)
            #pool_sema.release()
            test_queue.task_done()      
        

