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
python textAtt.py tasks=[fit] data=debate data.batch_size=128 trainer.max_epochs=1
```
If all goes well the following output should be produced:
```
Fitting on fold 0
GPU available: True, used: True
[2021-02-15 00:18:38,580][lightning][INFO] - GPU available: True, used: True
TPU available: None, using: 0 TPU cores
[2021-02-15 00:18:38,580][lightning][INFO] - TPU available: None, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[2021-02-15 00:18:38,580][lightning][INFO] - LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name          | Type               | Params
-----------------------------------------------------
0 | multihead_att | MultiHeadAttention | 4.1 M 
1 | pool          | MaxPooling         | 0     
2 | linear        | Linear             | 1.5 K 
3 | softmax       | LogSoftmax         | 0     
4 | loss          | NLLLoss            | 0     
5 | f1_score      | F1                 | 0     
-----------------------------------------------------
4.1 M     Trainable params
0         Non-trainable params
4.1 M     Total params

Epoch 0: 100%|██████████████████████████████████████████████████████████████████████████| 24/24 [00:02<00:00,  8.24it/s, loss=2.54, v_num=8, val_loss=0.828, F1=0.283]

```