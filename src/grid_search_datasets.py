import logging

from train_base_bert import train_base_bert
from train_ower_bert import train_ower_bert
from util.util import get_default_args


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = get_default_args()

    dataset_choices = [
        ['ower-v4-cde-cde-100-1', 1, 1024],
        ['ower-v4-cde-irt-100-1', 1, 1024],
        ['ower-v4-cde-irt-100-5', 5, 1024],
        ['ower-v4-cde-irt-100-15', 15, 1024],
        ['ower-v4-cde-irt-100-30', 20, 1024],
        ['ower-v4-fb-irt-100-1', 1, 1024],
        ['ower-v4-fb-irt-100-5', 5, 1024],
        ['ower-v4-fb-irt-100-15', 15, 1024],
        ['ower-v4-fb-irt-100-30', 30, 1024],
        ['ower-v4-fb-owe-100-1', 1, 1024]
    ]

    for dataset_choice in dataset_choices:
        dataset, sent_count, batch_size = dataset_choice

        args.ower_dir = f'data/ower/{dataset}'
        args.sent_count = sent_count

        args.try_batch_size = True

        while True:
            args.batch_size = batch_size

            try:
                logging.info(f'Try batch size {batch_size} for dataset {dataset}')
                train_base_bert(args)

                logging.info(f'Works. Use batch size {batch_size} for dataset {dataset}')
                break

            except RuntimeError:
                logging.warning('Batch too large')
                batch_size //= 2
                dataset_choice[2] = batch_size

    print(dataset_choices)

    # for i in range(3):
    #
    #     for dataset, sent_count in dataset_choices:
    #         args.ower_dir = f'data/ower/{dataset}'
    #         args.sent_count = sent_count
    #
    #         args.log_dir = f'runs/datasets_base-bert_{dataset}_{i}'
    #         train_base_bert(args)
    #
    #         args.log_dir = f'runs/datasets_ower-bert_{dataset}_{i}'
    #         train_ower_bert(args)


if __name__ == '__main__':
    main()
