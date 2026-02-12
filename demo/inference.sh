CUDA_VISIBLE_DEVICES=8 python demo.py --input /data1/guankai/dataset/RealLQ/ADEChallengeData2016/images/validation/*.jpg --output /home/guankai/RASS/output/rass_reallq --config-file ../configs/ade20k/semantic-segmentation/maskformer2_RASS.yaml --opts MODEL.WEIGHTS /home/guankai/Mutual_Enhancement/SCR/preset/RAS.pth

