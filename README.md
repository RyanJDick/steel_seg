# steel_seg
ML model for steel defect segmentation (for the Severstal Steel Kaggle Competition).
sudo docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ryan/src/steel_seg:/src/steel_seg -p 8888:8888 -p 6006:6006 nvcr.io/nvidia/tensorflow:19.09-py3


TODO, in order:

https://github.com/qubvel/segmentation_models#simple-training-pipeline
- Tune on validation set, evaluate on test set
- I think the classification network is about as good as I can expect, confirm this with a train and test set
- Explore alternative loss functions:
    - Weight boundary pixels more heavily
    - BCE + dice loss
    - Jacquard?
    - Focal loss?
- Try to get MixedPrecision actually working - and write a blog post about it
- Dataset generator for non-patch segmentation for easy oversampling
- Retry training non-patch model? (With oversampling of classes)
- Cropping augmentations
- Get rid of old dataset class
