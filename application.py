import cfg, time
import cv2,logging
from data_processing_app import DataProcessingUnit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Application:
    def __init__(self):
        cv2.setUseOptimized(True)
        self.basedir = cfg.basedir
        self.outputdir = cfg.outputdir
        self.obj_dpu = DataProcessingUnit()

    def dataProcessingUnit(self,front,side):
        status, frontrcnnmask, sidercnnmask, frontmaskall, sidemaskall = self.obj_dpu.process_maskrcnn(front, side)
        if cfg.debug == 'True':
            logger.info('Mask Rcnn Successfull',time.time()-t1 )

        return frontrcnnmask, sidercnnmask, status