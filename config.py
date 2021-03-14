DATASET_DIR = "/mnt/d/Datasets"
CHECKPOINT_DIR = "./checkpoints"

hparams = {
    "ADAM_lr": 10e-4,
    "batch_size": 3,
    "SGD_lr": 10e-5,
    "SGD_l2_penalty": 1e-5,
    "weights_init_a": -0.05,
    "weights_init_b": 0.05,
    "epochs": 10,
    "activation": "maxout"
}
