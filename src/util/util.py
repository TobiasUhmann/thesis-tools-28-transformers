import logging
from argparse import Namespace, ArgumentParser
from typing import Dict

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('ower_dir', metavar='ower-dir',
                        help='Path to (input) OWER Directory')

    parser.add_argument('class_count', metavar='class-count', type=int,
                        help='Number of classes distinguished by the classifier')

    parser.add_argument('sent_count', metavar='sent-count', type=int,
                        help='Number of sentences per entity')

    device_choices = ['cpu', 'cuda']
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', choices=device_choices, default=default_device,
                        help='Where to perform tensor operations, one of {} (default: {})'.format(
                            device_choices, default_device))

    default_batch_size = 4
    parser.add_argument('--batch-size', dest='batch_size', type=int, metavar='INT', default=default_batch_size,
                        help='Batch size (default: {})'.format(default_batch_size))

    default_epoch_count = 20
    parser.add_argument('--epoch-count', dest='epoch_count', type=int, metavar='INT', default=default_epoch_count,
                        help='Number of training epochs (default: {})'.format(default_epoch_count))

    default_log_dir = None
    parser.add_argument('--log-dir', dest='log_dir', metavar='STR', default=default_log_dir,
                        help='Tensorboard log directory (default: {})'.format(default_log_dir))

    parser.add_argument('--log-steps', dest='log_steps', action='store_true',
                        help='Log after steps, otherwise log after epochs')

    default_learning_rate = 1e-5
    parser.add_argument('--lr', dest='lr', type=float, metavar='FLOAT', default=default_learning_rate,
                        help='Learning rate (default: {})'.format(default_learning_rate))

    default_sent_len = 64
    parser.add_argument('--sent-len', dest='sent_len', type=int, metavar='INT', default=default_sent_len,
                        help='Sentence length short sentences are padded and long sentences cropped to'
                             ' (default: {})'.format(default_sent_len))

    parser.add_argument('--try-batch-size', dest='try_batch_size', action='store_true',
                        help='Try to perform a single train and valid loop to see whether the batch_size is ok')

    args = parser.parse_args()

    ## Log applied config

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ower-dir', args.ower_dir))
    logging.info('    {:24} {}'.format('class-count', args.class_count))
    logging.info('    {:24} {}'.format('sent-count', args.sent_count))
    logging.info('    {:24} {}'.format('--batch-size', args.batch_size))
    logging.info('    {:24} {}'.format('--device', args.device))
    logging.info('    {:24} {}'.format('--epoch-count', args.epoch_count))
    logging.info('    {:24} {}'.format('--log-dir', args.log_dir))
    logging.info('    {:24} {}'.format('--log-steps', args.log_steps))
    logging.info('    {:24} {}'.format('--lr', args.lr))
    logging.info('    {:24} {}'.format('--sent-len', args.sent_len))
    logging.info('    {:24} {}'.format('--try-batch-size', args.try_batch_size))

    return args


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
    args.log_dir = None
    args.log_steps = True

    return args


def log_class_metrics(data: Dict, writer: SummaryWriter, x: int, class_count: int) -> None:
    """
    Calculate class-wise metrics and log metrics of most/least common metrics to Tensorboard

    :param data: {'train': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]},
                  'valid': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]}}

    :param class_count: Log <class_count> most common and <class_count> least common classes
    """

    most_common_classes = range(0, 3)
    least_common_classes = range(class_count - 3, class_count)
    log_classes = [*most_common_classes, *least_common_classes]

    for split, metrics in data.items():
        prfs_list = precision_recall_fscore_support(metrics['gt_classes_stack'],
                                                    metrics['pred_classes_stack'],
                                                    average=None,
                                                    zero_division=0)

        # c = class
        for c, (prec, rec, f1, _), in enumerate(zip(*prfs_list)):
            if c not in log_classes:
                continue

            writer.add_scalars('precision', {f'{split}_{c}': prec}, x)
            writer.add_scalars('recall', {f'{split}_{c}': rec}, x)
            writer.add_scalars('f1', {f'{split}_{c}': f1}, x)


def log_macro_metrics(data: Dict, writer: SummaryWriter, x: int) -> None:
    """
    Calculate macro metrics across all classes and log them to Tensorboard

    :param data: {'train': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]},
                  'valid': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]}}

    :param x: Value on x-axis
    """

    for split, metrics in data.items():
        prec, rec, f1, _ = precision_recall_fscore_support(metrics['gt_classes_stack'],
                                                           metrics['pred_classes_stack'],
                                                           average='macro',
                                                           zero_division=0)

        writer.add_scalars('precision', {split: prec}, x)
        writer.add_scalars('recall', {split: rec}, x)
        writer.add_scalars('f1', {split: f1}, x)
