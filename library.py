import torch

print(torch.__version__)           # 1.9.0
print(torch.version.cuda)           # 11.1
print(torch.cuda.is_available())     #True


# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
# https://pytorch.org/docs/stable/notes/cuda.html
print(torch.cuda.empty_cache())
# https://pytorch.org/docs/stable/notes/cuda.html

import gc
del variables
gc.collect()