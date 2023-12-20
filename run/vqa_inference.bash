
name=$2
image_name = $3


output=snap/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --infer_image image_name --tiny --test test --valid "" --single_infer True --load snap/vqa/vqa_lx/BEST \
    --tqdm --output $output ${@:3}



    
