import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

import base64

from param import args
from utils_old import load_obj_tsv




# The path to data and image features.
TEST_IMG_ROOT = 'sample_inference/'
VQA_DATA_ROOT = 'sample_inference/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}

TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


class VQADataset:

    def __init__(self, splits: str,single_infer = False):
        self.name = splits
        self.splits = splits.split(',')

        
        self.data = []
        if(single_infer):
            self.data.extend(json.load(open(TEST_IMG_ROOT+"test_single.json")))
        else:
            for split in self.splits:
                self.data.extend(json.load(open("data/vqa/%s.json" % split)))
            print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)



class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset,single_infer = False):
        super().__init__()
        self.raw_dataset = dataset

        self.single_infer = single_infer

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []

        

        for split in dataset.splits:

            print("HEYYYYYYY")
            print(split)

            print(self.single_infer)
            if split == 'test':
                print('breaking out')
                break
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv(
                os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                topk=load_topk))

        # Convert img list to dict
        #if split == test and self.single_infer is True:

        #self.imgid2img = {}
        ##   self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features

        self.data = []
        print(self.single_infer)
        if self.single_infer:
            for datum in self.raw_dataset.data:
                if True:
                    self.data.append(datum)
            print("Use %d data in torch dataset" % (len(self.data)))
            print()
        else:

            for datum in self.raw_dataset.data:
                if datum['img_id'] in self.imgid2img:
                    self.data.append(datum)
            print("Use %d data in torch hehe dataset" % (len(self.data)))
            print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info

        if(self.single_infer):
            f = open(os.path.join(TEST_IMG_ROOT ,'test_single.tsv'))
            img_info = f.readline()
            f.close()
            img_info = img_info.split('\t')
            decode_img = self._decodeIMG(img_info)
            img_h = decode_img[0]
            img_w = decode_img[1]
            feats = decode_img[-1].copy()
            boxes = decode_img[-2].copy()
            del decode_img


        else:

            img_info = self.imgid2img[img_id]
            obj_num = img_info['num_boxes']
            feats = img_info['features'].copy()
            boxes = img_info['boxes'].copy()
            assert obj_num == len(boxes) == len(feats)

            img_h, img_w = img_info['img_h'], img_info['img_w']


        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques

    def _decodeIMG(self, img_info):



        img_h = int(img_info[1])
        img_w = int(img_info[2])
        boxes = img_info[-2]
        boxes = np.frombuffer(base64.b64decode(boxes), dtype=np.float32)
        boxes = boxes.reshape(36,4)
        boxes.setflags(write=False)
        feats = img_info[-1]
        feats = np.frombuffer(base64.b64decode(feats), dtype=np.float32)
        feats = feats.reshape(36,-1)
        feats.setflags(write=False)
        return [img_h, img_w, boxes, feats]

class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset


    def dump_result(self, quesid2ans: dict, path):
  
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)


