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

    # args.ower_dir
    args.class_count = 100
    # args.sent_count

    # args.batch_size
    args.device = 'cuda'
    args.epoch_count = 20
    # args.log_dir
    args.log_steps = True
    args.lr = 1e-5
    # args.save_dir
    args.sent_len = 64

    # Datasets with respective sentence counts and appropriate batch sizes.
    # Start out with twice the batch size that works on a GTX 1080 Ti with 11GB RAM.
    # [[(dataset, sent count), BASE batch size, OWER batch size]]
    dataset_model_choices = [
        [('ower-v4-cde-cde-100-1', 1), 'base-bert', 256],
        [('ower-v4-cde-irt-100-1', 1), 'base-bert', 256],
        [('ower-v4-cde-irt-100-5', 5), 'base-bert', 64],
        [('ower-v4-cde-irt-100-15', 15), 'base-bert', 16],
        [('ower-v4-cde-irt-100-30', 30), 'base-bert', 16],
        [('ower-v4-fb-irt-100-1', 1), 'base-bert', 256],
        [('ower-v4-fb-irt-100-5', 5), 'base-bert', 64],
        [('ower-v4-fb-irt-100-15', 15), 'base-bert', 16],
        [('ower-v4-fb-irt-100-30', 30), 'base-bert', 8],
        [('ower-v4-fb-owe-100-1', 1), 'base-bert', 512],

        [('ower-v4-cde-cde-100-1', 1), 'ower-bert', 256],
        [('ower-v4-cde-irt-100-1', 1), 'ower-bert', 256],
        [('ower-v4-cde-irt-100-5', 5), 'ower-bert', 64],
        [('ower-v4-cde-irt-100-15', 15), 'ower-bert', 16],
        [('ower-v4-cde-irt-100-30', 30), 'ower-bert', 16],
        [('ower-v4-fb-irt-100-1', 1), 'ower-bert', 256],
        [('ower-v4-fb-irt-100-5', 5), 'ower-bert', 64],
        [('ower-v4-fb-irt-100-15', 15), 'ower-bert', 16],
        [('ower-v4-fb-irt-100-30', 30), 'ower-bert', 8],
        [('ower-v4-fb-owe-100-1', 1), 'ower-bert', 512]
    ]

    #
    # Try batches sizes. Decrease if graphics RAM is not sufficient until it fits.
    #

    args.log_dir = 'runs/try_batch_size/'
    args.save_dir = 'models/try_batch_size/'
    args.try_batch_size = True

    for dataset_model_choice in dataset_model_choices:
        (dataset, sent_count), model, batch_size = dataset_model_choice

        args.ower_dir = f'data/ower/{dataset}/'
        args.sent_count = sent_count

        args.model = model

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

    args.try_batch_size = False

    for i in range(3):
        for (dataset, sent_count), model, batch_size in dataset_model_choices:
            args.ower_dir = f'data/ower/{dataset}/'
            args.sent_count = sent_count

            args.batch_size = batch_size
            args.log_dir = f'runs/datasets_{model}_{dataset}_{i}/'
            args.model = model
            args.save_dir = f'models/datasets_{model}_{dataset}_{i}/'

            train(args)


if __name__ == '__main__':
    main()
