lr = 2e-4
epochs =6000
lamda1 = 1
lamda2 = 2

# Train:
train_batch_size = 2
cropsize = 256

betas = (0.5, 0.999)
weight_decay = 1e-5
factor = 0.1
patience = 10

# Val:
val_batch_size = 1  
cropsize_val = 256
shuffle_val = False

# JPEG压缩
compress=False
quality=80

# Dataset
TRAIN_PATH='/home/pod/shared-nvme/div2k/train'
# VAL_PATH = '/home/pod/shared-nvme/data10000/test'
VAL_PATH = '/home/pod/shared-nvme/div2k/val'
format_train = 'png'
format_val = 'png'
# format_val = 'png'

# Saving checkpoints:
HMODEL_PATH = '/home/pod/shared-nvme/StegTransx/checkpoint4'
RMODEL_PATH = '/home/pod/shared-nvme/StegTransx/checkpoint4'
HMODEL_PATH_100 = '/home/pod/shared-nvme/StegTransx/100_checkpoint'
RMODEL_PATH_100 = '/home/pod/shared-nvme/StegTransx/100_checkpoint'
# load_checkpoint
is_load=False
Hload='/home/pod/shared-nvme/StegTransx/checkpoint4/Hmodel.pth'
Rload='/home/pod/shared-nvme/StegTransx/checkpoint4/Rmodel.pth'

# Hload='/home/pod/shared-nvme/StegTransx/checkpoint_best/3/Hmodel.pth'
# Rload='/home/pod/shared-nvme/StegTransx/checkpoint_best/3/Rmodel.pth'

# Saving log
log_path='/home/pod/shared-nvme/StegTransx/logging/train_logg.txt'

# tensorboard
t_log='/home/pod/shared-nvme/StegTransx/tf-logs'

