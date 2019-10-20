# steel_seg

docker pull nvcr.io/nvidia/tensorflow:19.09-py3
docker build -t steel_seg .
docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ryan/src/steel_seg:/src/steel_seg -p 8888:8888 -p 6006:6006 steel_seg

ML model for steel defect segmentation (for the Severstal Steel Kaggle Competition).
sudo docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ryan/src/steel_seg:/src/steel_seg -p 8888:8888 -p 6006:6006 nvcr.io/nvidia/tensorflow:19.09-py3

pip install git+https://github.com/qubvel/segmentation_models


TODO, in order:

https://github.com/qubvel/segmentation_models#simple-training-pipeline
- Explore alternative loss functions:
    - BCE + focal
    - Weight boundary pixels more heavily
    - BCE + dice loss
    - Jacquard?
    - Focal loss?
- Try to get MixedPrecision actually working - and write a blog post about it
- Retry training non-patch model? (With oversampling of classes)
- Cropping augmentations
- Get rid of old dataset class

- ensemble of multiple models (maybe same model trained on different folds?)
- If I go back to patches, change the code to oversample more (it seems to still be getting too many empty patches)


20 - first imgaug model
21 - imgaug with out classifier
22 - imgaug without classifier, thresholds of [0.95, 0.95, 0.5, 0.5]
23 - imgaug with classifier, thresholds of [0.95, 0.95, 0.5, 0.5]
24 - new imgaug (extra night of training) with classifier, thresholds of [0.5, 0.5, 0.5, 0.5] for comparison against 20
###(25) - new imgaug (extra night of training) with classifier, thresholds of [0.5, 0.5, 0.5, 0.2] to try tuning the thresholds
26 - 24 + new classification model
27 - 26 + lower classification thresholds (0.7, 0.7, 0.7, 0.7)
28 - classification thresholds [0.7, 0.7, 0.9, 0.5]