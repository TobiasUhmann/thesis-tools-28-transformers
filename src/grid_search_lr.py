import logging
from argparse import Namespace
from pprint import pformat

from train import train


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    #
    # Specify config
    #

    args = Namespace()

    # args.samples_dir
    args.class_count = 100
    # args.sent_count
    # args.split_dir
    # args.texter_pkl
    # args.eval_yml

    # args.batch_size
    args.device = 'cuda'
    args.epoch_count = 50
    # args.log_dir
    args.log_steps = False
    # args.lr
    # args.model
    # args.overwrite
    args.random_seed = None
    args.sent_len = 64
    args.test = True
    # args.try_batch_size
    args.use_embs = 'cls'

    # Datasets with respective sentence counts and appropriate batch sizes.
    # Start out with twice the batch size that works on a GTX 1080 Ti with 11GB RAM.
    # [[(dataset, sent count), split, model, init batch size]]
    dataset_model_choices = [
        # Clean, Base
        [('cde-cde-1-clean', 1), 'cde-0', 'base', 256],
        [('cde-irt-1-clean', 1), 'cde-0', 'base', 256],
        [('cde-irt-5-clean', 5), 'cde-0', 'base', 64],
        [('cde-irt-15-clean', 15), 'cde-0', 'base', 16],
        [('cde-irt-30-clean', 30), 'cde-0', 'base', 16],
        [('fb-irt-1-clean', 1), 'fb-0', 'base', 256],
        [('fb-irt-5-clean', 5), 'fb-0', 'base', 64],
        [('fb-irt-15-clean', 15), 'fb-0', 'base', 16],
        [('fb-irt-30-clean', 30), 'fb-0', 'base', 8],
        [('fb-owe-1-clean', 1), 'fb-0', 'base', 512],

        # Clean, Power
        [('cde-cde-1-clean', 1), 'cde-0', 'power', 256],
        [('cde-irt-1-clean', 1), 'cde-0', 'power', 256],
        [('cde-irt-5-clean', 5), 'cde-0', 'power', 64],
        [('cde-irt-15-clean', 15), 'cde-0', 'power', 16],
        [('cde-irt-30-clean', 30), 'cde-0', 'power', 16],
        [('fb-irt-1-clean', 1), 'fb-0', 'power', 256],
        [('fb-irt-5-clean', 5), 'fb-0', 'power', 64],
        [('fb-irt-15-clean', 15), 'fb-0', 'power', 16],
        [('fb-irt-30-clean', 30), 'fb-0', 'power', 8],
        [('fb-owe-1-clean', 1), 'fb-0', 'power', 512],

        # Marked, Base
        [('cde-irt-1-marked', 1), 'cde-0', 'base', 256],
        [('cde-irt-5-marked', 5), 'cde-0', 'base', 64],
        [('cde-irt-15-marked', 15), 'cde-0', 'base', 16],
        [('cde-irt-30-marked', 30), 'cde-0', 'base', 16],
        [('fb-irt-1-marked', 1), 'fb-0', 'base', 256],
        [('fb-irt-5-marked', 5), 'fb-0', 'base', 64],
        [('fb-irt-15-marked', 15), 'fb-0', 'base', 16],
        [('fb-irt-30-marked', 30), 'fb-0', 'base', 8],

        # Marked, Power
        [('cde-irt-1-marked', 1), 'cde-0', 'power', 256],
        [('cde-irt-5-marked', 5), 'cde-0', 'power', 64],
        [('cde-irt-15-marked', 15), 'cde-0', 'power', 16],
        [('cde-irt-30-marked', 30), 'cde-0', 'power', 16],
        [('fb-irt-1-marked', 1), 'fb-0', 'power', 256],
        [('fb-irt-5-marked', 5), 'fb-0', 'power', 64],
        [('fb-irt-15-marked', 15), 'fb-0', 'power', 16],
        [('fb-irt-30-marked', 30), 'fb-0', 'power', 8],

        # Masked, Base
        [('cde-irt-1-masked', 1), 'cde-0', 'base', 256],
        [('cde-irt-5-masked', 5), 'cde-0', 'base', 64],
        [('cde-irt-15-masked', 15), 'cde-0', 'base', 16],
        [('cde-irt-30-masked', 30), 'cde-0', 'base', 16],
        [('fb-irt-1-masked', 1), 'fb-0', 'base', 256],
        [('fb-irt-5-masked', 5), 'fb-0', 'base', 64],
        [('fb-irt-15-masked', 15), 'fb-0', 'base', 16],
        [('fb-irt-30-masked', 30), 'fb-0', 'base', 8],

        # Masked, Power
        [('cde-irt-1-masked', 1), 'cde-0', 'power', 256],
        [('cde-irt-5-masked', 5), 'cde-0', 'power', 64],
        [('cde-irt-15-masked', 15), 'cde-0', 'power', 16],
        [('cde-irt-30-masked', 30), 'cde-0', 'power', 16],
        [('fb-irt-1-masked', 1), 'fb-0', 'power', 256],
        [('fb-irt-5-masked', 5), 'fb-0', 'power', 64],
        [('fb-irt-15-masked', 15), 'fb-0', 'power', 16],
        [('fb-irt-30-masked', 30), 'fb-0', 'power', 8],
    ]

    #
    # Try batches sizes. Decrease if graphics RAM is not sufficient until it fits.
    #

    for dataset_model_choice in dataset_model_choices:
        (dataset, sent_count), split_dir, model, batch_size = dataset_model_choice

        args.samples_dir = f'data/power/samples-v5/{dataset}/'
        args.sent_count = sent_count
        args.split_dir = f'data/power/split-v2/{split_dir}/'
        args.texter_pkl = f'data/power/texter-v2/try-batch-size.pkl'
        args.eval_yml = f'data/power/eval-v1/try-batch-size.yml'

        args.log_dir = 'runs/try_batch_size/'
        args.lr = 1e-5
        args.model = model
        args.overwrite = True
        args.try_batch_size = True

        while True:
            args.batch_size = batch_size

            try:
                logging.info(f'Try batch size {batch_size} for dataset {dataset} and model {model}.')
                train(args)

                # Halve once more, just to be safe
                batch_size //= 2
                dataset_model_choice[-1] = batch_size

                logging.info(f'Works. Use batch size {batch_size} for dataset {dataset} and model {model}.')
                break

            except RuntimeError:
                logging.warning(f'Batch size {batch_size} too large for dataset {dataset} and model {model}.'
                                f' Halve batch size to {batch_size // 2}.')

                batch_size //= 2
                dataset_model_choice[-1] = batch_size

    #
    # Log determined batch sizes
    #

    logging.info(f'dataset_model_choices =\n'
                 f'{pformat(dataset_model_choices)}')

    #
    # Perform grid search
    #

    for (dataset, sent_count), split_dir, model, batch_size in dataset_model_choices:
        for lr in [1e-4, 3e-5, 1e-5, 3e-6, 1e-7]:
            args.samples_dir = f'data/power/samples-v5/{dataset}/'
            args.sent_count = sent_count
            args.split_dir = f'data/power/split-v2/{split_dir}/'
            args.texter_pkl = f'data/power/texter-v2/final/{dataset}_{model}_{lr}.pkl'
            args.eval_yml = f'data/power/eval-v1/final/{dataset}_{model}_{lr}.yml'

            args.batch_size = batch_size
            args.log_dir = f'runs/final/{dataset}_{model}/'
            args.lr = lr
            args.model = model
            args.overwrite = False
            args.try_batch_size = False

            logging.info(f'Training on dataset {dataset} using model {model} with learning rate {lr}')
            train(args)


if __name__ == '__main__':
    main()
