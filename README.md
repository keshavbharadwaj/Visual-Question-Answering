
##########   Visual Question Answering (Deep Learning : Course Project)  ############


##################   Demo / Inference ####################

GPU is required for all tasks.

Note: only one sample can be infered at once. no batch inference. and please let me know if you are unable to run inference script. I am happy to give a demo 

Prerequisite: 
1. Navigate to working directory
2. create conda env, activate and do pip install -r requirements_vqa.txt
3. Download trained model from my drive link https://drive.google.com/file/d/12vpo3wsyMv670gkEgpQO2OA7Dwotggcv/view?usp=sharing and place it at snap/vqa/vqa_lx directory

Steps:
1. Place the sample images you want to test in sample_inference folder
2. edit the 'sent' field in the 'test_single.json' file.
3. in terminal run the below command by placing image name at '--infer_image' flag.

--->

PYTHONPATH=$PYTHONPATH:./src \
python src/tasks/vqa.py \
   --infer_image soccer.jpeg --tiny --test test --valid "" --single_infer True --load snap/vqa/vqa_lx/BEST \
   --tqdm --output sample_inference

<---

4. Please find the output answer for given input of image,question pair in 'test_predict.json' in the same sample_inference.json

##################   Training  ####################


Prerequisites:

1. Make sure the necessary dataset is downloaded for training.
mkdir -p data/vqa
wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/train.json -P data/vqa/
wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/nominival.json -P  data/vqa/
wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/minival.json -P data/vqa/

mkdir -p data/mscoco_imgfeat
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip

2. Download a minimal 3 task 1 epoch semi pre-trained model from my drive link https://drive.google.com/file/d/1NxVY6YSrbFuS3iVwh6HdgeoIiEdMwyRq/view?usp=sharing and place it in snap/pretrained

Steps:

1. Run command 'bash run/vqa_train.bash 0 vqa_lx
2. find the model at snap/vqa directory.

