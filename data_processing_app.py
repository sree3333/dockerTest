import os
import cv2
import gc
import numpy as np
import multiprocessing
from multiprocessing import Process
from data_processing_unit.maskrcnn import Maskrcnn



class DataProcessingUnit:
    def __init__(self):
        cv2.setUseOptimized(True)
        self.obj_maskrcnn = Maskrcnn()

    def process_maskrcnn(self, front, side):
        try:
            manager = multiprocessing.Manager()
            return_dict = manager.dict()

            def worker_process(return_dic, front, side):
                frontrcnnmask, frontmask = self.obj_maskrcnn.run_maskrcnn(front, 'f')
                sidercnnmask, sidemask = self.obj_maskrcnn.run_maskrcnn(side, 's')
                gc.collect()
                return_dic['1'] = True
                return_dic['2'] = frontrcnnmask
                return_dic['3'] = sidercnnmask
                return_dic['4'] = frontmask
                return_dic['5'] = sidemask

            p = Process(target=worker_process, args=(return_dict, front, side))
            p.start()
            p.join()
            return return_dict.values()

        except:
            return False, None, None, None, None

