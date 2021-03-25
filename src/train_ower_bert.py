import logging
from pathlib import Path
from typing import List, Tuple

import numpy
from torch import Tensor, tensor
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import DistilBertTokenizer, AdamW

from util.util import parse_args, log_class_metrics, log_macro_metrics
from data.ower.ower_dir import OwerDir
from data.ower.samples_tsv import Sample
from models.ower_bert import OwerBert


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()

    train_ower_bert(args)


def train_ower_bert(args):
    ower_dir_path = args.ower_dir
    class_count = args.class_count
    sent_count = args.sent_count

    batch_size = args.batch_size
    device = args.device
    epoch_count = args.epoch_count
    log_dir = args.log_dir
    log_steps = args.log_steps
    lr = args.lr
    sent_len = args.sent_len

    ## Check that (input) OWER Directory exists

    ower_dir = OwerDir(Path(ower_dir_path))
    ower_dir.check()

    ## Create model and tokenizer

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    classifier = OwerBert(class_count, sent_count)

    ## Load datasets and create dataloaders

    train_set = ower_dir.train_samples_tsv.load(class_count, sent_count)
    valid_set = ower_dir.valid_samples_tsv.load(class_count, sent_count)

    def generate_batch(batch: List[Sample]) -> Tuple[Tensor, Tensor, Tensor]:
        _, _, classes_batch, texts_batch = zip(*batch)

        flat_text_batch = [text for texts in texts_batch for text in texts]

        # extended_classes_batch = [[classes] * sent_count for classes in classes_batch]
        # stretched_classes_batch = [classes for extended_classes in extended_classes_batch for classes in
        #                            extended_classes]

        encoded = tokenizer(flat_text_batch, padding=True, truncation=True, max_length=sent_len, return_tensors='pt')

        flat_sent_batch = encoded.input_ids
        flat_mask_batch = encoded.attention_mask

        sents_batch = flat_sent_batch.reshape(len(batch), sent_count, -1)
        masks_batch = flat_mask_batch.reshape(len(batch), sent_count, -1)

        return sents_batch, masks_batch, tensor(classes_batch)

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=generate_batch, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=generate_batch)

    ## Calc class weights

    _, _, train_classes_stack, _ = zip(*train_set)
    train_classes_stack = numpy.array(train_classes_stack)
    train_freqs = train_classes_stack.mean(axis=0)

    class_weights = tensor(1 / train_freqs).to(device)

    ## Train

    classifier = classifier.to(device)

    criterion = BCEWithLogitsLoss(pos_weight=class_weights)
    # optimizer = Adam(bert.parameters(), lr=lr)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    writer = SummaryWriter(log_dir=log_dir)

    train_sample_idx = 0
    valid_sample_idx = 0

    for epoch in range(epoch_count):

        epoch_metrics = {
            'train': {'loss': 0.0, 'pred_classes_stack': [], 'gt_classes_stack': []},
            'valid': {'loss': 0.0, 'pred_classes_stack': [], 'gt_classes_stack': []}
        }

        ## Train

        classifier.train()

        for sents_batch, masks_batch, gt_batch in tqdm(train_loader):
            train_sample_idx += len(sents_batch)

            sents_batch = sents_batch.to(device)
            masks_batch = masks_batch.to(device)
            gt_batch = gt_batch.to(device)

            logits_batch = classifier(sents_batch, masks_batch)
            loss = criterion(logits_batch, gt_batch.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## Log metrics

            pred_batch = (logits_batch > 0).int()

            step_loss = loss.item()
            step_pred_batch = pred_batch.cpu().numpy().tolist()
            step_gt_batch = gt_batch.cpu().numpy().tolist()

            if log_steps:
                writer.add_scalars('loss', {'train': step_loss}, train_sample_idx)

                step_metrics = {'train': {
                    'pred_classes_stack': step_pred_batch,
                    'gt_classes_stack': step_gt_batch
                }}

                log_class_metrics(step_metrics, writer, train_sample_idx, class_count)
                log_macro_metrics(step_metrics, writer, train_sample_idx)

            else:
                epoch_metrics['train']['loss'] += step_loss
                epoch_metrics['train']['pred_classes_stack'] += step_pred_batch
                epoch_metrics['train']['gt_classes_stack'] += step_gt_batch

        ## Validate

        classifier.eval()

        for sents_batch, masks_batch, gt_batch in tqdm(valid_loader):
            valid_sample_idx += len(sents_batch)

            sents_batch = sents_batch.to(device)
            masks_batch = masks_batch.to(device)
            gt_batch = gt_batch.to(device)

            logits_batch = classifier(sents_batch, masks_batch)
            loss = criterion(logits_batch, gt_batch.float())

            ## Log metrics

            pred_batch = (logits_batch > 0).int()

            step_loss = loss.item()
            step_pred_batch = pred_batch.cpu().numpy().tolist()
            step_gt_batch = gt_batch.cpu().numpy().tolist()

            if log_steps:
                writer.add_scalars('loss', {'valid': step_loss}, valid_sample_idx)

                step_metrics = {'valid': {
                    'pred_classes_stack': step_pred_batch,
                    'gt_classes_stack': step_gt_batch
                }}

                log_class_metrics(step_metrics, writer, valid_sample_idx, class_count)
                log_macro_metrics(step_metrics, writer, valid_sample_idx)

            else:
                epoch_metrics['valid']['loss'] += step_loss
                epoch_metrics['valid']['pred_classes_stack'] += step_pred_batch
                epoch_metrics['valid']['gt_classes_stack'] += step_gt_batch

        if not log_steps:
            ## Log loss

            train_loss = epoch_metrics['train']['loss'] / len(train_loader)
            valid_loss = epoch_metrics['valid']['loss'] / len(valid_loader)

            writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch)

            ## Log metrics

            log_class_metrics(epoch_metrics, writer, epoch, class_count)
            log_macro_metrics(epoch_metrics, writer, epoch)


if __name__ == '__main__':
    main()
