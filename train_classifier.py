import logging
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle
from typing import List, Tuple

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import Tensor, tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

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

        for sents in sents_batch:
            shuffle(sents)

        contexts_batch = [' '.join(sents) for sents in sents_batch]

        encoded_batch = tokenizer(contexts_batch, padding=True, truncation=True, max_length=sent_len,
                                  return_tensors='pt')

        return encoded_batch, tensor(classes_batch)

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=generate_batch, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=generate_batch)

    ## Train

    criterion = BCEWithLogitsLoss()
    optimizer = Adam(bert.parameters(), lr=lr)

    bert = bert.to(device)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epoch_count):

        ## Train

        train_loss = 0.0

        # Valid gt/pred classes across all batches
        train_gt_classes_stack: List[List[int]] = []
        train_pred_classes_stack: List[List[int]] = []

        bert.train()
        for step, (ctxt_batch, gt_batch) in enumerate(tqdm(train_loader)):
            input_ids = ctxt_batch.input_ids.to(device)
            attention_mask = ctxt_batch.attention_mask.to(device)
            gt_batch = gt_batch.to(device)

            pred_batch = bert(input_ids, attention_mask).logits

            loss = criterion(gt_batch.float(), pred_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_classes_batch = (pred_batch > 0).int()

            train_gt_classes_stack += gt_batch.cpu().numpy().tolist()
            train_pred_classes_stack += pred_classes_batch.cpu().numpy().tolist()

            train_loss += loss.item()

        ## Validate

        valid_loss = 0.0

        # Valid gt/pred classes across all batches
        valid_gt_classes_stack: List[List[int]] = []
        valid_pred_classes_stack: List[List[int]] = []

        bert.eval()
        for step, (ctxt_batch, gt_batch) in enumerate(tqdm(valid_loader)):
            input_ids = ctxt_batch.input_ids.to(device)
            attention_mask = ctxt_batch.attention_mask.to(device)
            gt_batch = gt_batch.to(device)

            outputs_batch = bert(input_ids, attention_mask).logits
            loss = criterion(outputs_batch, gt_batch.float())

            valid_loss += loss.item()

            pred_classes_batch = (outputs_batch > 0).int()

            valid_gt_classes_stack += gt_batch.cpu().numpy().tolist()
            valid_pred_classes_stack += pred_classes_batch.cpu().numpy().tolist()

        ## Log loss

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch)

        ## Log metrics for most/least common classes

        # tps = train precisions, vps = valid precisions, etc.
        tps = precision_score(train_gt_classes_stack, train_pred_classes_stack, average=None)
        vps = precision_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None)
        trs = recall_score(train_gt_classes_stack, train_pred_classes_stack, average=None)
        vrs = recall_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None)
        tfs = f1_score(train_gt_classes_stack, train_pred_classes_stack, average=None)
        vfs = f1_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None)

        # Log metrics for each class c
        for c, (tp, vp, tr, vr, tf, vf), in enumerate(zip(tps, vps, trs, vrs, tfs, vfs)):

            # many classes -> log only first and last ones
            if (class_count > 2 * 3) and (3 <= c <= len(tps) - 3 - 1):
                continue

            writer.add_scalars('precision', {f'train_{c}': tp}, epoch)
            writer.add_scalars('precision', {f'valid_{c}': vp}, epoch)
            writer.add_scalars('recall', {f'train_{c}': tr}, epoch)
            writer.add_scalars('recall', {f'valid_{c}': vr}, epoch)
            writer.add_scalars('f1', {f'train_{c}': tf}, epoch)
            writer.add_scalars('f1', {f'valid_{c}': vf}, epoch)

        ## Log macro metrics over all classes

        # mtp = mean train precision, mvp = mean valid precision, etc.
        mtp = tps.mean()
        mvp = vps.mean()
        mtr = trs.mean()
        mvr = vrs.mean()
        mtf = tfs.mean()
        mvf = vfs.mean()

        writer.add_scalars('precision', {'train': mtp}, epoch)
        writer.add_scalars('precision', {'valid': mvp}, epoch)
        writer.add_scalars('recall', {'train': mtr}, epoch)
        writer.add_scalars('recall', {'valid': mvr}, epoch)
        writer.add_scalars('f1', {'train': mtf}, epoch)
        writer.add_scalars('f1', {'valid': mvf}, epoch)


if __name__ == '__main__':
    main()
