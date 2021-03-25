import logging

from train_base_bert import train_base_bert
from train_ower_bert import train_ower_bert
from util.util import get_default_args


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = get_default_args()

    dataset_choices = [
        ('ower-v4-cde-cde-100-1', 1),
        ('ower-v4-cde-irt-100-1', 1),
        ('ower-v4-cde-irt-100-5', 5),
        ('ower-v4-cde-irt-100-15', 15),
        ('ower-v4-cde-irt-100-30', 20),
        ('ower-v4-fb-irt-100-1', 1),
        ('ower-v4-fb-irt-100-5', 5),
        ('ower-v4-fb-irt-100-15', 15),
        ('ower-v4-fb-irt-100-30', 30),
        ('ower-v4-fb-owe-100-1', 1)
    ]

    for i in range(3):

        for dataset, sent_count in dataset_choices:
            args.ower_dir = f'data/ower/{dataset}'
            args.sent_count = sent_count

            args.log_dir = f'runs/datasets_base-bert_{dataset}_{i}'
            train_base_bert(args)

            args.log_dir = f'runs/datasets_ower-bert_{dataset}_{i}'
            train_ower_bert(args)


if __name__ == '__main__':
    main()
