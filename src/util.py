from argparse import Namespace


def get_default_args():
    args = Namespace()

    # Input data
    args.ower_dir = 'data/ower/ower-v4-fb-irt-100-5/'
    args.class_count = 100
    args.sent_count = 5

    # Pre-processing
    args.sent_len = 64

    # Training
    args.batch_size = 4
    args.device = 'cuda'
    args.epoch_count = 20
    args.lr = 1e-5

    # Logging
    args.logdir = None
    args.log_steps = True

    return args
