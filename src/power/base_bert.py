from torch import Tensor
from torch.nn import Module
from transformers import DistilBertForSequenceClassification


class BaseBert(Module):
    bert: DistilBertForSequenceClassification

    def __init__(self, pre_trained: str, class_count: int):
        super().__init__()

        self.bert = DistilBertForSequenceClassification.from_pretrained(pre_trained, num_labels=class_count)

    def forward(self, tok_lists_batch: Tensor, masks_batch: Tensor) -> Tensor:
        """
        :param    tok_lists_batch:  IntTensor[batch_size, sent_count, sent_len]
        :param    masks_batch:      IntTensor[batch_size, sent_count, sent_len]
        :return:  logits_batch:     FloatTensor[batch_size, class_count]
        """

        # Flatten sentences and masks for BERT
        #
        # < tok_lists_batch:       IntTensor[batch_size, sent_count, sent_len]
        # < masks_batch:           IntTensor[batch_size, sent_count, sent_len]
        # > flat_tok_lists_batch:  IntTensor[batch_size * sent_count, sent_len]
        # > flat_masks_batch:      IntTensor[batch_size * sent_count, sent_len]

        batch_size, sent_count, sent_len = tok_lists_batch.shape

        flat_tok_lists_batch = tok_lists_batch.reshape(batch_size * sent_count, sent_len)
        flat_masks_batch = masks_batch.reshape(batch_size * sent_count, sent_len)

        # Embed and classify sentences
        #
        # < flat_tok_lists_batch:  IntTensor[batch_size * sent_count, sent_len]
        # < flat_masks_batch:      IntTensor[batch_size * sent_count, sent_len]
        # > flat_logits_batch:     FloatTensor[batch_size * sent_count, class_count]

        flat_logits_batch = self.bert(input_ids=flat_tok_lists_batch, attention_mask=flat_masks_batch).logits

        # Aggregate logits of same entity
        #
        # < flat_logits_batch:  FloatTensor[batch_size * sent_count, class_count]
        # > logits_batch:       FloatTensor[batch_size, class_count]

        _, class_count = flat_logits_batch.shape

        logits_batch = flat_logits_batch.reshape(batch_size, sent_count, class_count).mean(dim=1)

        return logits_batch
