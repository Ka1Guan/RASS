export DETECTRON2_DATASETS=/path/to/your/d2_datasets/

# train
python train_net.py --config-file configs/ade20k/semantic-segmentation/maskformer2_RASS.yaml --num-gpus 8

# eval
python train_net.py --config-file configs/ade20k/semantic-segmentation/maskformer2_RASS.yaml \
--eval-only MODEL.WEIGHTS preset/model/rass.pth \
OUTPUT_DIR ./experiments