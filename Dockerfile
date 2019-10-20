FROM nvcr.io/nvidia/tensorflow:19.09-py3

# Required for opencv-python
RUN apt-get update \
    && apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender-dev

COPY docker-requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
