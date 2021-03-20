import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Dict

import numpy
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor, tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW

from dao.ower.ower_dir import OwerDir
from dao.ower.samples_tsv import Sample


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()

    train_classifier(args)


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

    default_batch_size = 8
    parser.add_argument('--batch-size', dest='batch_size', type=int, metavar='INT', default=default_batch_size,
                        help='Batch size (default: {})'.format(default_batch_size))

    default_epoch_count = 20
    parser.add_argument('--epoch-count', dest='epoch_count', type=int, metavar='INT', default=default_epoch_count,
                        help='Number of training epochs (default: {})'.format(default_epoch_count))

    default_log_dir = None
    parser.add_argument('--log-dir', dest='log_dir', metavar='STR', default=default_log_dir,
                        help='Tensorboard log directory (default: {})'.format(default_log_dir))

    default_learning_rate = 0.01
    parser.add_argument('--lr', dest='lr', type=float, metavar='FLOAT', default=default_learning_rate,
                        help='Learning rate (default: {})'.format(default_learning_rate))

    model_choices = ['base', 'ower']
    default_model = 'ower'
    parser.add_argument('--model', dest='model', choices=model_choices, default=default_model)

    default_sent_len = 128
    parser.add_argument('--sent-len', dest='sent_len', type=int, metavar='INT', default=default_sent_len,
                        help='Sentence length short sentences are padded and long sentences cropped to'
                             ' (default: {})'.format(default_sent_len))

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
    logging.info('    {:24} {}'.format('--lr', args.lr))
    logging.info('    {:24} {}'.format('--model', args.model))
    logging.info('    {:24} {}'.format('--sent-len', args.sent_len))

    return args


def train_classifier(args):
    ower_dir_path = args.ower_dir
    class_count = args.class_count
    sent_count = args.sent_count

    batch_size = args.batch_size
    device = args.device
    epoch_count = args.epoch_count
    log_dir = args.log_dir
    lr = args.lr
    model = args.model
    sent_len = args.sent_len

    ## Check that (input) OWER Directory exists

    ower_dir = OwerDir(Path(ower_dir_path))
    ower_dir.check()

    ## Create model and tokenizer

    model_name = 'distilbert-base-uncased'
    bert = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=class_count)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    ## Load datasets and create dataloaders

    train_set = ower_dir.train_samples_tsv.load(class_count, sent_count)
    valid_set = ower_dir.valid_samples_tsv.load(class_count, sent_count)

    def generate_batch(batch: List[Sample]) -> Tuple[Tensor, Tensor]:
        _, _, classes_batch, sents_batch = zip(*batch)

        stretched_sent_batch = [sent for sents in sents_batch for sent in sents]

        extended_classes_batch = [[classes] * sent_count for classes in classes_batch]
        stretched_classes_batch = [classes for extended_classes in extended_classes_batch for classes in extended_classes]

        encoded_batch = tokenizer(stretched_sent_batch, padding=True, truncation=True, max_length=sent_len,
                                  return_tensors='pt')

        return encoded_batch, tensor(stretched_classes_batch)

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=generate_batch, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=generate_batch)

    ##

    _, _, train_classes_stack, _ = zip(*train_set)
    train_classes_stack = numpy.array(train_classes_stack)
    train_freqs = train_classes_stack.mean(axis=0)

    class_weights = tensor(1 / train_freqs).to(device)

    ## Train

    bert = bert.to(device)

    criterion = BCEWithLogitsLoss(pos_weight=class_weights)
    # optimizer = Adam(bert.parameters(), lr=lr)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epoch_count):

        metrics = {
            'train': {'loss': 0.0, 'pred_classes_stack': [], 'gt_classes_stack': []},
            'valid': {'loss': 0.0, 'pred_classes_stack': [], 'gt_classes_stack': []}
        }

        ## Train

        bert.train()

        for step, (ctxt_batch, gt_classes_batch) in enumerate(tqdm(train_loader)):
            metrics = {
                'train': {'loss': 0.0, 'pred_classes_stack': [], 'gt_classes_stack': []},
                'valid': {'loss': 0.0, 'pred_classes_stack': [], 'gt_classes_stack': []}
            }

            input_ids_batch = ctxt_batch.input_ids.to(device)
            attention_mask_batch = ctxt_batch.attention_mask.to(device)
            gt_classes_batch = gt_classes_batch.to(device)

            outputs_batch = bert(input_ids_batch, attention_mask_batch).logits
            loss = criterion(outputs_batch, gt_classes_batch.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            track_metrics(metrics['train'], loss, outputs_batch, gt_classes_batch)

            ## Log loss

            writer.add_scalars('loss', {'train': metrics['train']['loss']}, step)

            ## Log metrics

            log_class_metrics(metrics, writer, step, class_count)
            log_macro_metrics(metrics, writer, step)

        ## Validate

        bert.eval()

        for ctxt_batch, gt_classes_batch in tqdm(valid_loader):
            input_ids_batch = ctxt_batch.input_ids.to(device)
            attention_mask_batch = ctxt_batch.attention_mask.to(device)
            gt_classes_batch = gt_classes_batch.to(device)

            outputs_batch = bert(input_ids_batch, attention_mask_batch).logits
            loss = criterion(outputs_batch, gt_classes_batch.float())

            track_metrics(metrics['valid'], loss, outputs_batch, gt_classes_batch)

        # ## Log loss
        #
        # train_loss = metrics['train']['loss'] / len(train_loader)
        # valid_loss = metrics['valid']['loss'] / len(valid_loader)
        #
        # writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch)
        #
        # ## Log metrics
        #
        # log_class_metrics(metrics, writer, epoch, class_count)
        # log_macro_metrics(metrics, writer, epoch)


def track_metrics(metrics: Dict, loss: Tensor, outputs_batch: Tensor, gt_classes_batch: Tensor) -> None:
    """
    :param metrics: {'loss': loss, 'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]}
    """

    pred_classes_batch = (outputs_batch > 0).int()

    metrics['loss'] += loss.item()
    metrics['pred_classes_stack'] += pred_classes_batch.cpu().numpy().tolist()
    metrics['gt_classes_stack'] += gt_classes_batch.cpu().numpy().tolist()


def log_class_metrics(metrics: Dict, writer: SummaryWriter, epoch: int, class_count: int) -> None:
    """
    Calculate class-wise metrics and log metrics of most/least common metrics to Tensorboard

    :param metrics: {'train': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]},
                     'valid': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]}}

    :param class_count: Log <class_count> most common and <class_count> least common classes
    """

    train_prfs_list = precision_recall_fscore_support(metrics['train']['gt_classes_stack'],
                                                      metrics['train']['pred_classes_stack'],
                                                      average=None)

    valid_prfs_list = precision_recall_fscore_support(metrics['valid']['gt_classes_stack'],
                                                      metrics['valid']['pred_classes_stack'],
                                                      average=None)

    most_common_classes = range(0, 3)
    least_common_classes = range(class_count - 3, class_count)
    log_classes = [*most_common_classes, *least_common_classes]

    # c = class, tp = train precision, tr = train recall, etc.
    for c, (tp, tr, tf, _, vp, vr, vf, _), in enumerate(zip(*train_prfs_list, *valid_prfs_list)):
        if c not in log_classes:
            continue

        writer.add_scalars('precision', {f'train_{c}': tp}, epoch)
        writer.add_scalars('precision', {f'valid_{c}': vp}, epoch)
        writer.add_scalars('recall', {f'train_{c}': tr}, epoch)
        writer.add_scalars('recall', {f'valid_{c}': vr}, epoch)
        writer.add_scalars('f1', {f'train_{c}': tf}, epoch)
        writer.add_scalars('f1', {f'valid_{c}': vf}, epoch)


def log_macro_metrics(metrics: Dict, writer: SummaryWriter, epoch: int) -> None:
    """
    Calculate macro metrics across all classes and log them to Tensorboard

    :param metrics: {'train': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]},
                     'valid': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]}}
    """

    tp, tr, tf, _ = precision_recall_fscore_support(metrics['train']['gt_classes_stack'],
                                                    metrics['train']['pred_classes_stack'],
                                                    average='macro')

    vp, vr, vf, _ = precision_recall_fscore_support(metrics['valid']['gt_classes_stack'],
                                                    metrics['valid']['pred_classes_stack'],
                                                    average='macro')

    writer.add_scalars('precision', {'train': tp}, epoch)
    writer.add_scalars('precision', {'valid': vp}, epoch)
    writer.add_scalars('recall', {'train': tr}, epoch)
    writer.add_scalars('recall', {'valid': vr}, epoch)
    writer.add_scalars('f1', {'train': tf}, epoch)
    writer.add_scalars('f1', {'valid': vf}, epoch)


if __name__ == '__main__':
    main()
