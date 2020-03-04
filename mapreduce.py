
import os
import argparse
from threading import Thread

import argparse

from yolov3.utils.utils import load_classes


class GenericInputData(object):
    def read(self):
        raise NotImplementedError 
    
    @classmethod
    def generate_inputs(cls, config):
        raise NotImplementedError 

class PathInputData(GenericInputData):
    def __init__(self, path):
        super().__init__()
        self.path = path
        
    def read(self):
        return open(self.path).read()

    @classmethod
    def generate_inputs(cls, config):
        data_dir = config['data_dir']
    
        for name in os.listdir(data_dir):
            yield cls(os.path.join(data_dir, name))

class GenericWorker(object):
    def __init__(self, input_data, config):
        self.input_data = input_data
        self.config = config 
        self.result = None
        

    def map(self):
        raise NotImplementedError

    def reduce(self):
        raise NotImplementedError
    
    @classmethod 
    def create_workers(cls, input_class, config):
        workers = []
        for input_data in input_class.generate_inputs(config):
            workers.append(cls(input_data, config))
        
        return workers

class StatisticsWorker(GenericWorker):
    def __init__(self, input_data, config):
        super().__init__(input_data, config)

    
    def map(self):
        data = self.input_data.read()
        clss_ind = self.config['clss']

        lines = data.split('\n')
        self.result = 0
        for line in lines: 
            if line == '':
                continue
            
            clss = int(float((line.strip().split(',')[-1])))
            
            if clss == clss_ind:
                self.result += 1

    def reduce(self, other):
        self.result += other.result
          

def execute(workers):
    
    threads = [Thread(target=w.map) for w in workers]
    
    for thread in threads: thread.start()
    for thread in threads: thread.join()

    first, rest = threads[0], threads[1:]
    
    for worker in rest:
        first.reduce(worker)
    
    return first.result
    
def mapreduce(input_class, worker_class, config):
    
    workers = worker_class.create_workers(input_class, config)
    return execute(workers)    

def load_classes_local(class_path):
    fp = open(class_path, 'r')

    data = fp.read()
    classes = data.split('\n')
    classes = [clss.strip() for clss in classes]

    return classes

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--class_path', required=True)
    parser.add_argument('--clss', required=True)

    args = parser.parse_args()
    
    classes = load_classes(args.class_path)
    
    args.clss = args.clss.strip().lower()
    
    try:
        clss_ind = classes.index(args.clss)
    except ValueError:
        print('{} is not in the predefined classes'.format(args.clss))
        exit(1)

    config = {}
    config['data_dir'] = args.label_dir 
    config['clss'] = clss_ind

    sum_objects = mapreduce(PathInputData, StatisticsWorker, config)
    
    print("number of objects is: {}".format(sum_objects))
   

