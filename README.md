# Text-Att

### 1. Quick Start

```shell script
# clone the project 
git clone git@github.com:celsofranssa/Text-Att.git

# change directory to project folder
cd Text-Att/

# Create a new virtual environment by choosing a Python interpreter 
# and making a ./venv directory to hold it:
virtualenv -p python3 ./venv

# activate the virtual environment using a shell-specific command:
source ./venv/bin/activate

# install dependecies
pip install -r requirements.txt

# setting python path
export PYTHONPATH=$PATHONPATH:<path-to-project-dir>/Text-Att/

# (if you need) to exit virtualenv later:
deactivate
```

### 2. Test Run
The following bash command fits Text-Att model over debate dataset using batch_size=64 and a single epoch.
```
python main.py \
  tasks=[fit,predict,eval] \
  model=Self-Att \
  data=WEBKB \
  data.batch_size=64 \
  data.num_workers=12
```
If all goes well the following output should be produced:
```
Fitting Self-Att over REUT (fold 9) with fowling params:

    model:
      name: Self-Att
      att_encoder:
        _target_: source.encoder.MultiHeadAttentionEncoder.MultiHeadAttentionEncoder
        hidden_size: 768
        num_heads: 12
        dropout: 0.1
        pooling:
          _target_: source.pooling.MaxPooling.MaxPooling
      num_classes: 9
      hidden_size: 768
      lr: 5.0e-05
      weight_decay: 0.01
    
    data:
      name: WEBKB
      dir: resource/dataset/webkb/
      folds:[0-9]
      max_length: 128
      num_classes: 90
      batch_size: 64
      num_workers: 12
    
    ...
    
    tasks: [fit,predict,eval]
    
    trainer:
      max_epochs: 16
      gpus: 1
      patience: 7
      min_delta: 0.01
      precision: 32

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name         | Type                      | Params
-----------------------------------------------------------
0 | att_encoder  | MultiHeadAttentionEncoder | 4.1 M 
1 | cls_head     | Sequential                | 69.2 K
2 | loss         | CrossEntropyLoss          | 0     
3 | val_metrics  | MetricCollection          | 0     
4 | test_metrics | MetricCollection          | 0     
-----------------------------------------------------------
4.2 M     Trainable params
0         Non-trainable params
4.2 M     Total params
16.813    Total estimated model params size (MB)
Epoch 15: 100%|█████████████████████████████████████████████████████████████████████████████| 187/187 [00:01<00:00, 111.35it/s, loss=0.801, v_num=0, val_Mac-F1=0.456, val_Mic-F1=0.776, val_Wei-F1=0.819]
```

### 2. Some Results