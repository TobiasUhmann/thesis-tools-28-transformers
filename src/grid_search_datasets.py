import logging
from argparse import Namespace
from pprint import pformat

from train_base_bert import train_base_bert
from train_ower_bert import train_ower_bert


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    ## Specify config

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
    args.sent_len = 64

    # Datasets with respective sentence counts and appropriate batch sizes.
    # Start out with batch sizes that work on a GTX 1080 Ti with 11GB RAM.
    # [[(dataset, sent count), BASE batch size, OWER batch size]]
    dataset_model_choices = [
        [('ower-v4-cde-cde-100-1', 1), 256, 256],
        [('ower-v4-cde-irt-100-1', 1), 256, 256],
        [('ower-v4-cde-irt-100-5', 5), 64, 64],
        [('ower-v4-cde-irt-100-15', 15), 16, 16],
        [('ower-v4-cde-irt-100-30', 30), 16, 16],
        [('ower-v4-fb-irt-100-1', 1), 256, 256],
        [('ower-v4-fb-irt-100-5', 5), 64, 64],
        [('ower-v4-fb-irt-100-15', 15), 16, 16],
        [('ower-v4-fb-irt-100-30', 30), 16, 16],
        [('ower-v4-fb-owe-100-1', 1), 512, 512]
    ]

    ## Try batches sizes. Decrease if graphics RAM is not sufficient until it fits.
    
    for dataset_model_choice in dataset_model_choices:
        (dataset, sent_count), base_batch_size, ower_batch_size = dataset_model_choice

        args.ower_dir = f'data/ower/{dataset}'
        args.sent_count = sent_count

        args.log_dir = None
        args.try_batch_size = True

        while True:
            args.batch_size = base_batch_size

            try:
                logging.info(f'Try batch size {base_batch_size} for dataset {dataset} and model BASE-BERT.')
                train_base_bert(args)

                logging.info(f'Works. Use batch size {base_batch_size} for dataset {dataset} and model BASE-BERT.')
                break

            except RuntimeError:
                logging.warning(f'Batch size {base_batch_size} too large for dataset {dataset} and model BASE-BERT.'
                                f' Half batch size to {base_batch_size // 2}.')

                base_batch_size //= 2
                dataset_model_choice[2] = base_batch_size

        while True:
            args.batch_size = ower_batch_size

            try:
                logging.info(f'Try batch size {ower_batch_size} for dataset {dataset} and model OWER-BERT.')
                train_ower_bert(args)

                logging.info(f'Works. Use batch size {ower_batch_size} for dataset {dataset} and model OWER-BERT.')
                break

            except RuntimeError:
                logging.warning(f'Batch size {ower_batch_size} too large for dataset {dataset} and model OWER-BERT.'
                                f' Half batch size to {ower_batch_size // 2}.')

                ower_batch_size //= 2
                dataset_model_choice[3] = ower_batch_size

    ## Log determined batch sizes

    logging.info('')
    logging.info(f'dataset_model_choices =\n'
                 f'{pformat(dataset_model_choices)}')

    ## Perform grid search

    for i in range(3):

        for (dataset, sent_count), base_batch_size, ower_batch_size in dataset_model_choices:
            args.ower_dir = f'data/ower/{dataset}'
            args.sent_count = sent_count

            args.batch_size = base_batch_size
            args.log_dir = f'runs/datasets_base-bert_{dataset}_{i}'
            train_base_bert(args)

            args.batch_size = ower_batch_size
            args.log_dir = f'runs/datasets_ower-bert_{dataset}_{i}'
            train_ower_bert(args)


if __name__ == '__main__':
    main()
