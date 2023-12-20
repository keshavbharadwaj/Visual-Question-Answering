import PIL.Image
import io
import torch
import numpy as np
from processing_image import Preprocess

from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils

import pandas as pd

import base64


def im2feat(image_path):


    URL = image_path
    img = PIL.Image.open(URL)
  
# get width and height
    width = img.width
    height = img.height

    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

    image_preprocess = Preprocess(frcnn_cfg)

    images, sizes, scales_yx = image_preprocess(URL)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )


    features = output_dict.get("roi_features").view(-1).cpu().detach().numpy()
    normalized_boxes = output_dict.get("normalized_boxes").view(-1).cpu().detach().numpy()

    feat = base64.b64encode(features)
    boxes = base64.b64encode(normalized_boxes)

    feat = base64.b64encode(features)
    boxes = base64.b64encode(normalized_boxes)



    with open('sample_inference/test_single.tsv', 'w') as f:
        f.write('COCO_777' + "\t" + str(height) + "\t" + str(width) + "\t" + str(boxes)[2:-1] + "\t" + str(feat)[2:-1] + "\n")

 


def im2feat_trail():

    URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg"
    #URL = "https://vqa.cloudcv.org/media/test2014/COCO_test2014_000000262567.jpg"
    #OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
    #ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
    #VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"

    URL = 'pic1.jpg'
    img = PIL.Image.open(URL)
  
# get width and height
    width = img.width
    height = img.height


    # for visualizing output
    def showarray(a, fmt="jpeg"):
        a = np.uint8(np.clip(a, 0, 255))
        f = io.BytesIO()
        PIL.Image.fromarray(a).save(f, fmt)
        display(Image(data=f.getvalue()))


    #objids = utils.get_data(OBJ_URL)
    #attrids = utils.get_data(ATTR_URL)
    #vqa_answers = utils.get_data(VQA_URL)

    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

    image_preprocess = Preprocess(frcnn_cfg)

    print(frcnn_cfg.max_detections)
    images, sizes, scales_yx = image_preprocess(URL)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )


    #print(output_dict)

    features = output_dict.get("roi_features").view(-1).cpu().detach().numpy()

    normalized_boxes = output_dict.get("normalized_boxes").view(-1).cpu().detach().numpy()

    feat = base64.b64encode(features)


    boxes = base64.b64encode(normalized_boxes)

    feat = base64.b64encode(features)


    boxes = base64.b64encode(normalized_boxes)
    #feats = np.frombuffer(base64.b64decode(feat), dtype=np.float32)
    #boxess = np.frombuffer(base64.b64decode(boxes), dtype=np.float32)

    
    print(features.shape)
    #print(feats.shape)

    #print(normalized_boxes.shape)
    #print(boxess.shape)
    #print(np.array_equal(features, feats))
    #print(np.array_equal(normalized_boxes, boxess))

    print(type(feat))
   # dic = {'img_name':["COCO_777"],'height':[height],'width':[width],'boxes':[boxes] ,'feat':[str(feat)[2:-1]]  }
    #df = pd.DataFrame(dic)
    #x = T.view(2,3)



    #print(df.head())
    #print(features.shape)
    #print(normalized_boxes.shape)

    with open('sample.tsv', 'w') as f:
        f.write('COCO_777' + "\t" + str(height) + "\t" + str(width) + "\t" + str(boxes)[2:-1] + "\t" + str(feat)[2:-1] + "\n")

    #df.to_csv('sample.tsv', sep='\t', index=False,header = False)



#im2feat()