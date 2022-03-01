# activate venv and set Python path
source ~/projects/venvs/Text-Att/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/Text-Att/

# REUT
python main.py \
  tasks=[fit,predict,eval] \
  model=Self-Att \
  data=REUT \
  data.batch_size=64 \
  data.num_workers=12

# WEBKB
python main.py \
  tasks=[fit,predict,eval] \
  model=Self-Att \
  data=WEBKB \
  data.batch_size=64 \
  data.num_workers=12