# Text classifier

### Overview
> This project is tensorflow implementation text classification and embedding vector extraction.

### Installation

#### Dependencies
This project requires:
* [Python](https://www.python.org/downloads/release/python-380/) (>= 3.8)
* [Numpy](https://pypi.org/project/numpy/1.20.1/) (>= 1.20.1)
* [Tensorflow](https://pypi.org/project/tensorflow/2.7.0/) (>= 2.7.0)
* Others


### Usage

##### Preprocess and create tfrecord
example:

    python main.py --op=preprocess \
                   --data_dir=data \
                   --config_dir=configs \
                   --chunk_size=1000 \
                   --split_ratio=0.3, 0.1


#### Train model
example:

    python main.py --op=train \
                   --data_dir=data \
                   --ckpt_dir=checkpoints \
                   --config_dir=configs \
                   --logs_dir=logs \
                   --batch_size=128 \
                   --epochs=1000 \
                   --lr=5e-5 \
                   --gpus=2 \
                   --use_xla=True

#### Inference
example:

    python main.py --op=predict \
                   --data_dir=data \
                   --ckpt_dir=checkpoints \
                   --config_dir=config

#### Visualize log using tensorboard
example:

    tensorboard --logdir={target_dir} \
                --host=0.0.0.0
                --port 6006

### License
* Apache License 2.0
