import logging
import random
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle
from typing import List, Tuple, Dict

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor, tensor
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW

from data.power.eval_yml import EvalYml
from data.power.samples.samples_dir import SamplesDir
from data.power.samples.samples_tsv import Sample
from data.power.split.split_dir import SplitDir
from data.power.texter_pkl import TexterPkl
from models.ent import Ent
from models.rel import Rel
from power.base import Base
from power.texter import Texter


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    train(args)

    logging.info('Finished successfully')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('samples_dir', metavar='samples-dir',
                        help='Path to (input) Power Samples Directory')

    parser.add_argument('class_count', metavar='class-count', type=int,
                        help='Number of classes distinguished by the classifier')

    parser.add_argument('sent_count', metavar='sent-count', type=int,
                        help='Number of sentences per entity')

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) Power Split Directory')

    parser.add_argument('texter_pkl', metavar='texter-pkl',
                        help='Path to (output) Power Texter PKL')

    parser.add_argument('eval_yml', metavar='eval_yml',
                        help='Path to (output) Power Eval YML')

    default_batch_size = 4
    parser.add_argument('--batch-size', dest='batch_size', type=int, metavar='INT', default=default_batch_size,
                        help='Batch size (default: {})'.format(default_batch_size))

    device_choices = ['cpu', 'cuda']
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', choices=device_choices, default=default_device,
                        help='Where to perform tensor operations, one of {} (default: {})'.format(
                            device_choices, default_device))

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

    model_choices = ['base', 'power']
    default_model_choice = 'power'
    parser.add_argument('--model', dest='model', choices=model_choices, default=default_model_choice,
                        help='One of {} (default: {})'.format(model_choices, default_model_choice))

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    default_sent_len = 64
    parser.add_argument('--sent-len', dest='sent_len', type=int, metavar='INT', default=default_sent_len,
                        help='Sentence length short sentences are padded and long sentences cropped to'
                             ' (default: {})'.format(default_sent_len))

    parser.add_argument('--test', dest='test', action='store_true',
                        help='Evaluate on test data after training')

    parser.add_argument('--try-batch-size', dest='try_batch_size', action='store_true',
                        help='Try to perform a single train and valid loop to see whether the batch_size is ok')

    default_use_embs = 'cls'
    use_embs_choices = ['cls', 'no-cls', 'mask', 'all']
    parser.add_argument('--use-embs', dest='use_embs', choices=use_embs_choices, default=default_use_embs,
                        help="Which of BERT's token embeddings to gather sentence embedding, one of {}"
                             " (default: {})".format(use_embs_choices, default_use_embs))

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('samples-dir', args.samples_dir))
    logging.info('    {:24} {}'.format('class-count', args.class_count))
    logging.info('    {:24} {}'.format('sent-count', args.sent_count))
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))
    logging.info('    {:24} {}'.format('texter-pkl', args.texter_pkl))
    logging.info('    {:24} {}'.format('eval-yml', args.eval_yml))
    logging.info('    {:24} {}'.format('--batch-size', args.batch_size))
    logging.info('    {:24} {}'.format('--device', args.device))
    logging.info('    {:24} {}'.format('--epoch-count', args.epoch_count))
    logging.info('    {:24} {}'.format('--log-dir', args.log_dir))
    logging.info('    {:24} {}'.format('--log-steps', args.log_steps))
    logging.info('    {:24} {}'.format('--lr', args.lr))
    logging.info('    {:24} {}'.format('--model', args.model))
    logging.info('    {:24} {}'.format('--overwrite', args.overwrite))
    logging.info('    {:24} {}'.format('--random-seed', args.random_seed))
    logging.info('    {:24} {}'.format('--sent-len', args.sent_len))
    logging.info('    {:24} {}'.format('--test', args.test))
    logging.info('    {:24} {}'.format('--try-batch-size', args.try_batch_size))
    logging.info('    {:24} {}'.format('--use-embs', args.use_embs))

    return args


def train(args):
    samples_dir_path = args.samples_dir
    class_count = args.class_count
    sent_count = args.sent_count
    split_dir_path = args.split_dir
    texter_pkl_path = args.texter_pkl
    eval_yml_path = args.eval_yml

    batch_size = args.batch_size
    device = args.device
    epoch_count = args.epoch_count
    log_dir = args.log_dir
    log_steps = args.log_steps
    lr = args.lr
    model_name = args.model
    overwrite = args.overwrite
    sent_len = args.sent_len
    test = args.test
    try_batch_size = args.try_batch_size
    use_embs = args.use_embs

    #
    # Check that (input) Power Samples Directory exists
    #

    logging.info('Check that (input) Power Samples Directory exists ...')

    samples_dir = SamplesDir(Path(samples_dir_path))
    samples_dir.check()

    #
    # Check that (input) Power Split Directory exists
    #

    logging.info('Check that (input) Power Split Directory exists ...')

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Check that (output) Power Texter PKL does not exist
    #

    logging.info('Check that (output) Power Texter PKL does not exist ...')

    texter_pkl = TexterPkl(Path(texter_pkl_path))

    if not overwrite:
        texter_pkl.check(should_exist=False)

    #
    # Check that (output) Power Eval YML does not exist
    #

    logging.info('Check that (output) Power Eval YML does not exist ...')

    eval_yml = EvalYml(Path(eval_yml_path))

    if not overwrite:
        eval_yml.check(should_exist=False)

    #
    # Load entity/relation labels
    #

    logging.info('Load entity/relation labels ...')

    ent_to_lbl = split_dir.entities_tsv.load()
    rel_to_lbl = split_dir.relations_tsv.load()

    #
    # Create model
    #

    rel_tail_freq_lbl_tuples = samples_dir.classes_tsv.load()

    classes = [(Rel(rel, rel_to_lbl[rel]), Ent(tail, ent_to_lbl[tail]))
               for rel, tail, _, _ in rel_tail_freq_lbl_tuples]

    pre_trained = 'distilbert-base-uncased'

    if model_name == 'base':
        model = Base(pre_trained, classes).to(device)
    elif model_name == 'power':
        model = Texter(pre_trained, classes, use_embs=use_embs).to(device)
    else:
        raise ValueError('Invalid model "{}". Must be one of {}.'.format(model_name, ['base', 'power']))

    #
    # Load datasets and create dataloaders
    #

    logging.info('Load datasets and create dataloaders ...')

    train_set = samples_dir.train_samples_tsv.load(class_count, sent_count)
    valid_set = samples_dir.valid_samples_tsv.load(class_count, sent_count)
    test_set = samples_dir.test_samples_tsv.load(class_count, sent_count)

    def generate_batch(batch: List[Sample]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        :param    batch:            [Sample(ent, ent_lbl, [class], [sent])]

        :return:  ent_batch:        IntTensor[batch_size],
                  tok_lists_batch:  IntTensor[batch_size, sent_count, sent_len],
                  masks_batch:      IntTensor[batch_size, sent_count, sent_len],
                  classes_batch:    IntTensor[batch_size, class_count]
        """

        ent_batch, _, classes_batch, sents_batch = zip(*batch)

        for sents in sents_batch:
            shuffle(sents)

        flat_sents_batch = [sent for sents in sents_batch for sent in sents]

        encoded = model.tokenizer(flat_sents_batch, padding=True, truncation=True, max_length=sent_len,
                                   return_tensors='pt')

        b_size = len(ent_batch)  # usually b_size == batch_size, except for last batch in samples
        tok_lists_batch = encoded.input_ids.reshape(b_size, sent_count, -1)
        masks_batch = encoded.attention_mask.reshape(b_size, sent_count, -1)

        return tensor(ent_batch), tok_lists_batch, masks_batch, tensor(classes_batch)

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=generate_batch, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=generate_batch)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=generate_batch)

    #
    # Calc class weights
    #

    logging.info('Calc class weights ...')

    _, _, train_classes_stack, _ = zip(*train_set)
    train_freqs = np.array(train_classes_stack).mean(axis=0)

    class_weights = tensor(1 / train_freqs)

    #
    # Prepare training
    #

    logging.info('Prepare training ...')

    model = model.to(device)

    criterion = BCEWithLogitsLoss(pos_weight=class_weights.to(device))

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    writer = SummaryWriter(log_dir=log_dir)

    #
    # Train
    #

    logging.info('Train ...')

    best_valid_f1 = 0

    # Global progress for Tensorboard
    train_steps = 0
    valid_steps = 0

    for epoch in range(epoch_count):

        epoch_metrics = {
            'train': {'loss': 0.0, 'pred_classes_stack': [], 'gt_classes_stack': []},
            'valid': {'loss': 0.0, 'pred_classes_stack': [], 'gt_classes_stack': []}
        }

        #
        # Train
        #

        model.train()

        for _, sents_batch, masks_batch, gt_batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            train_steps += len(sents_batch)

            sents_batch = sents_batch.to(device)
            masks_batch = masks_batch.to(device)
            gt_batch = gt_batch.to(device).float()

            logits_batch = model(sents_batch, masks_batch)[0]
            loss = criterion(logits_batch, gt_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #
            # Log metrics
            #

            pred_batch = (logits_batch > 0).int()

            step_loss = loss.item()
            step_pred_batch = pred_batch.cpu().numpy().tolist()
            step_gt_batch = gt_batch.cpu().numpy().tolist()

            epoch_metrics['train']['loss'] += step_loss
            epoch_metrics['train']['pred_classes_stack'] += step_pred_batch
            epoch_metrics['train']['gt_classes_stack'] += step_gt_batch

            if log_steps:
                writer.add_scalars('loss', {'train': step_loss}, train_steps)

                step_metrics = {'train': {
                    'pred_classes_stack': step_pred_batch,
                    'gt_classes_stack': step_gt_batch
                }}

                log_class_metrics(step_metrics, writer, train_steps, class_count)
                log_macro_metrics(step_metrics, writer, train_steps)

            if try_batch_size:
                break

        #
        # Validate
        #

        model.eval()

        for _, sents_batch, masks_batch, gt_batch in tqdm(valid_loader, desc=f'Epoch {epoch}'):
            valid_steps += len(sents_batch)

            sents_batch = sents_batch.to(device)
            masks_batch = masks_batch.to(device)
            gt_batch = gt_batch.to(device).float()

            logits_batch = model(sents_batch, masks_batch)[0]
            loss = criterion(logits_batch, gt_batch)

            #
            # Log metrics
            #

            pred_batch = (logits_batch > 0).int()

            step_loss = loss.item()
            step_pred_batch = pred_batch.cpu().numpy().tolist()
            step_gt_batch = gt_batch.cpu().numpy().tolist()

            epoch_metrics['valid']['loss'] += step_loss
            epoch_metrics['valid']['pred_classes_stack'] += step_pred_batch
            epoch_metrics['valid']['gt_classes_stack'] += step_gt_batch

            if log_steps:
                writer.add_scalars('loss', {'valid': step_loss}, valid_steps)

                step_metrics = {'valid': {
                    'pred_classes_stack': step_pred_batch,
                    'gt_classes_stack': step_gt_batch
                }}

                log_class_metrics(step_metrics, writer, valid_steps, class_count)
                log_macro_metrics(step_metrics, writer, valid_steps)

            if try_batch_size:
                break

        #
        # Log loss
        #

        train_loss = epoch_metrics['train']['loss'] / len(train_loader)
        valid_loss = epoch_metrics['valid']['loss'] / len(valid_loader)

        writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch)

        #
        # Log metrics
        #

        log_class_metrics(epoch_metrics, writer, epoch, class_count)
        valid_f1 = log_macro_metrics(epoch_metrics, writer, epoch)

        #
        # Persist Texter
        #

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            texter_pkl.save(model)

        if try_batch_size:
            break

    #
    # Load best Texter and test
    #

    if test:
        model = texter_pkl.load()
        model.eval()

        test_progress = 0
        test_metrics = {
            'test': {'loss': 0.0, 'pred_stack': [], 'gt_stack': []}
        }

        for _, sents_batch, masks_batch, gt_batch in tqdm(valid_loader, desc='Test'):
            test_progress += len(sents_batch)

            sents_batch = sents_batch.to(device)
            masks_batch = masks_batch.to(device)
            gt_batch = gt_batch.to(device).float()

            logits_batch = model(sents_batch, masks_batch)[0]
            loss = criterion(logits_batch, gt_batch)

            #
            # Track test metrics
            #

            pred_batch = (logits_batch > 0).int()

            step_loss = loss.item()
            step_pred_batch = pred_batch.cpu().numpy().tolist()
            step_gt_batch = gt_batch.cpu().numpy().tolist()

            test_metrics['test']['loss'] += step_loss
            test_metrics['test']['pred_stack'] += step_pred_batch
            test_metrics['test']['gt_stack'] += step_gt_batch

        #
        # Calc and persist test metrics
        #

        test_loss = test_metrics['test']['loss'] / len(test_loader)
        logging.info(f'Test Loss = {test_loss}')

        prec, rec, f1, _ = precision_recall_fscore_support(test_metrics['test']['gt_stack'],
                                                           test_metrics['test']['pred_stack'],
                                                           average='macro',
                                                           zero_division=0)

        eval_yml.save({'precision': f'{prec:.4f}',
                       'recall': f'{rec:.4f}',
                       'f1': f'{f1:.4f}'})


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


def log_macro_metrics(data: Dict, writer: SummaryWriter, x: int) -> float:
    """
    Calculate macro metrics across all classes and log them to Tensorboard

    :param data: {'train': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]},
                  'valid': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]}}

    :param x: Value on x-axis

    :return F1 score
    """

    for split, metrics in data.items():
        prec, rec, f1, _ = precision_recall_fscore_support(metrics['gt_classes_stack'],
                                                           metrics['pred_classes_stack'],
                                                           average='macro',
                                                           zero_division=0)

        writer.add_scalars('precision', {split: prec}, x)
        writer.add_scalars('recall', {split: rec}, x)
        writer.add_scalars('f1', {split: f1}, x)

    return f1


if __name__ == '__main__':
    main()
