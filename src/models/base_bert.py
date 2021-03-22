import torch
from torch import Tensor
from torch.nn import Module
from transformers import DistilBertForSequenceClassification


class BaseBert(Module):

    bert: DistilBertForSequenceClassification

    def __init__(self, class_count: int):
        super().__init__()

        self.bert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=class_count)

    def forward(self, tok_lists_batch: Tensor) -> Tensor:
        """
        :param tok_lists_batch: (batch_size, sent_count, sent_len)
        :return: (batch_size, class_count)
        """

        # Average entity's sentences to a single context
        #
        # < tok_lists_batch  (batch_size, sent_count, sent_len)
        # > ctxt_batch       (batch_size, sent_len)

        batch_size, _, sent_len = tok_lists_batch.shape

        ctxt_batch = torch.randint(1000, (batch_size, sent_len)).cuda()

        # Push context through BERT
        #
        # < ctxt_batch    (batch_size, emb_size)
        # > logits_batch  (batch_size, class_count)

        logits_batch = self.bert(ctxt_batch).logits

        return logits_batch
