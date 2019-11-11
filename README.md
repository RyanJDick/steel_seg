# steel_seg

ML model for steel defect segmentation (for the Severstal Steel Kaggle Competition).

```
docker pull nvcr.io/nvidia/tensorflow:19.09-py3
docker build -t steel_seg .
docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ryan/src/steel_seg:/src/steel_seg -p 8888:8888 -p 6006:6006 steel_seg
```

## Things learned from the competition
- Use off-the-shelf pretrained models as much as possible (for vision tasks). You will probably not be able to outperform these models with a custom model trained from scratch.
- Prioritize tools for EDA and model evaluation. This is important.
- Learn as much as you can about the data. For example, in this competition, there were clusters of similar images im the public training dataset that did not appear in the test set.
- Prioritize being able to evaluate quickly.
    - Consider using Kaggle's command-line tools, and uploading your submission as a dataset to avoid having to run a slow Kaggle kernel (and upload your model)
- Do not get sucked into overfitting on the public dataset. There was significant shakeup when the models were run on the private dataset.
- Don't bother with tf.data. Use numpy as much as possible. This makes it much easier to iterate on preprocessing techniques, customize example sampling etc.
- Create a data directory separate from source directory for data, checkpoints, and everything else that is large and will never be committed (cleaner git workspace, and allows code editor to scan workspace more efficiently)
