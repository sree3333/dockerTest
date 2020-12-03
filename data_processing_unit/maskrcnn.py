import cv2, numpy as np
import torch
from torchvision import transforms as T
from data_processing_unit.utils import getPersonHeight

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.config import cfg as mcfg
import cfg,os

class Maskrcnn():
    CATEGORIES = [
        "__background",
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
        "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
        "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
        "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
        "chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
        "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
    ]

    def __init__(self):
        self.outputdir = cfg.outputdir + '/maskrcnn'
        os.system('mkdir -p ' + self.outputdir)
        confidence_threshold = 0.8
        show_mask_heatmaps = False
        masks_per_dim = 1
        min_image_size = 224
        mcfg.merge_from_file(cfg.maskrcnnpath)
        mcfg.freeze()
        self.cfg = mcfg.clone()
        self.model = build_detection_model(mcfg)
        self.model.eval()
        self.device = torch.device(mcfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        checkpointer = DetectronCheckpointer(mcfg, self.model)
        _ = checkpointer.load(mcfg.MODEL.WEIGHT)
        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.4
        self.masker = Masker(threshold=mask_threshold, padding=100)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        cfg = self.cfg
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def addborder(self, edge):
        meshheight = getPersonHeight(edge)
        r = 2000.0 / float(meshheight)
        height, width = edge.shape[:2]
        edge = cv2.resize(edge, (int(r * width), 2000))
        h, w = edge.shape
        edge = cv2.copyMakeBorder(edge, int(h * 0.02), int(h * 0.02),
                        int(w * 0.02), int(w * 0.02), cv2.BORDER_CONSTANT)
        return edge

    def run_maskrcnn(self, image,pose):
        if pose == 'f':
            predictions = self.compute_prediction(image)
            top_predictions = self.select_top_predictions(predictions)
            result = image.copy()
            maskimg,maskimgall = self.overlay_mask(result, top_predictions)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # maskimg = cv2.dilate(maskimg, kernel)
            # maskimgall = cv2.dilate(maskimgall, kernel)

            if cfg.debug == 'True':
                cv2.imwrite(self.outputdir + '/front.jpg', maskimg)
            return maskimg, maskimgall
        if pose == 's':
            predictions = self.compute_prediction(image)
            top_predictions = self.select_top_predictions(predictions)
            result = image.copy()
            maskimg, maskimgall = self.overlay_mask(result, top_predictions)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            maskimg = cv2.dilate(maskimg, kernel)
            maskimgall = cv2.dilate(maskimgall, kernel)
            if cfg.debug == 'True':
                cv2.imwrite(self.outputdir + '/side.jpg', maskimg)
            return maskimg, maskimgall

    def compute_prediction(self, original_image):
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]
        prediction = predictions[0]
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        if prediction.has_field("mask"):
            masks = prediction.get_field("mask")
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def overlay_mask(self, image, predictions):
        labelscolor = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labelscolor]
        boxes = predictions.bbox
        masks = predictions.get_field("mask").numpy()
        maxh = 0
        fmask = None
        maskimg = np.zeros(image.shape[:2], np.uint8)
        maskimgall = np.zeros(image.shape[:2], np.uint8)
        for mask, box, label in zip(masks, boxes, labels):
            if label=='person' and maxh < (box[3] - box[1]):
                maxh = (box[3] - box[1])
                fmask = mask
            if label == 'person':
                threshall = mask[0, :, :, None]
                threshall = threshall.astype('uint8')
                contours, hierarchy = cv2.findContours(threshall, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                maskimgall = cv2.drawContours(maskimgall, contours, -1, [255], -1)

        thresh = fmask[0, :, :, None]
        thresh = thresh.astype('uint8')
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        maskimg = cv2.drawContours(maskimg, contours, -1, [255], -1)
        maskimgall = cv2.bitwise_xor(maskimg,maskimgall)
        return maskimg,maskimgall