from mpiigaze.config.config_node import ConfigNode

config = ConfigNode()

config.dataset = ConfigNode()
config.dataset.dataset_dir = ''

config.model = ConfigNode()
config.model.name = 'resnet_preact'

config.train = ConfigNode()
config.train.checkpoint = ''
config.train.resume = False
config.train.device = 'cuda'
config.train.batch_size = 64
config.train.optimizer = 'sgd'
config.train.base_lr = 0.01
config.train.momentum = 0.9
config.train.nesterov = True
config.train.weight_decay = 1e-4
config.train.start_epoch = 0
config.train.seed = 0
config.train.val_first = True
config.train.val_period = 1

config.train.test_id = 0
config.train.val_ratio = 0.1

config.train.outdir = 'results'
config.train.log_period = 100
config.train.checkpoint_period = 10

config.train.use_tensorboard = True
config.tensorboard = ConfigNode()
config.tensorboard.train_images = False
config.tensorboard.val_images = False
config.tensorboard.model_params = False

# optimizer
config.optim = ConfigNode()
# Adam
config.optim.adam = ConfigNode()
config.optim.adam.betas = (0.9, 0.999)
# AdaBound
config.optim.adabound = ConfigNode()
config.optim.adabound.betas = (0.9, 0.999)
config.optim.adabound.final_lr = 0.1
config.optim.adabound.gamma = 1e-3

# scheduler
config.scheduler = ConfigNode()
config.scheduler.epochs = 40
# warm up (options: none, linear, exponential)
config.scheduler.warmup = ConfigNode()
config.scheduler.warmup.type = 'none'
config.scheduler.warmup.epochs = 0
config.scheduler.warmup.start_factor = 1e-3
config.scheduler.warmup.exponent = 4
# main scheduler (options: constant, linear, multistep, cosine, sgdr)
config.scheduler.type = 'multistep'
config.scheduler.milestones = [20, 30]
config.scheduler.lr_decay = 0.1
config.scheduler.lr_min_factor = 0.001
config.scheduler.T0 = 10
config.scheduler.T_mul = 1.

# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.num_workers = 2
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = False

# test config
config.test = ConfigNode()
config.test.checkpoint = ''
config.test.outdir = ''
config.test.device = 'cuda'
config.test.batch_size = 256
# test data loader
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False

# cuDNN
config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False


def get_default_config():
    return config.clone()
