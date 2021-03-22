import torch
from torch import Tensor
from torch.nn import Module, Parameter, Softmax
from transformers import DistilBertModel


class OwerBert(Module):
    sent_count: int

    bert: DistilBertModel
    class_embs: Parameter
    multi_weight: Parameter
    multi_bias: Parameter

    def __init__(self, class_count: int, sent_count: int):
        super().__init__()

        self.sent_count = sent_count

        pre_trained = 'distilbert-base-uncased'
        self.bert = DistilBertModel.from_pretrained(pre_trained)

        emb_size = self.bert.config.dim
        self.class_embs = Parameter(torch.randn(class_count, emb_size))
        self.multi_weight = Parameter(torch.randn(class_count, emb_size))
        self.multi_bias = Parameter(torch.randn(class_count))

    def forward(self, toks_batch: Tensor, masks_batch: Tensor) -> Tensor:
        """
        :param toks_batch: (batch_size, sent_count, sent_len)
        :param masks_batch: (batch_size, sent_count, sent_len)
        :return (batch_size, class_count)
        """

        # Flatten sents_batch and masks_batch for BERT
        #
        # < toks_batch       (batch_size, sent_count, sent_len)
        # < masks_batch      (batch_size, sent_count, sent_len)
        # > flat_tok_batch   (batch_size * sent_count, sent_len)
        # > flat_mask_batch  (batch_size * sent_count, sent_len)

        batch_size, sent_count, sent_len = toks_batch.shape

        flat_tok_batch = toks_batch.reshape(batch_size * sent_count, sent_len)
        flat_mask_batch = masks_batch.reshape(batch_size * sent_count, sent_len)

        # Embed sentences
        #
        # < flat_tok_batch   (batch_size * sent_count, sent_len)
        # < flat_mask_batch  (batch_size * sent_count, sent_len)

        flat_sent_batch = self.bert(input_ids=flat_tok_batch, attention_mask=flat_mask_batch)\
            .last_hidden_state.mean(dim=1)

        # Restore batch shape
        #
        # < flat_sent_batch  (batch_size * sent_count, emb_size)
        # > sents_batch      (batch_size, sent_count, emb_size)

        _, emb_size = flat_sent_batch.shape

        sents_batch = flat_sent_batch.reshape(batch_size, sent_count, emb_size)

        # Calculate sent-class attentions
        #
        # < sents_batch      (batch_size, sent_count, emb_size)
        # < self.class_embs  (class_count, emb_size)
        # > atts_batch       (batch_size, class_count, sent_count)

        atts_batch = torch.einsum('bse, ce -> bcs', sents_batch, self.class_embs)

        # Softmax over sentences
        #
        # < atts_batch   (batch_size, class_count, sent_count)
        # > softs_batch  (batch_size, class_count, sent_count)

        softs_batch = Softmax(dim=-1)(atts_batch)

        # For each class, mix sentences according to attention
        #
        # < softs_batch  (batch_size, class_count, sent_count)
        # < sents_batch  (batch_size, sent_count, emb_size)
        # > mixes_batch  (batch_size, class_count, emb_size)

        mixes_batch = torch.bmm(softs_batch, sents_batch)

        # Push each mix throug its respective single-output linear layer,
        # i.e. scalar multiply each mix vector (of size <emb_size>) with
        # its respective weight vector (of size <emb_size>) and add the
        # bias afterwards.
        #
        # < mixes_batch        (batch_size, class_count, emb_size)
        # < self.multi_weight  (class_count, emb_size)
        # < self.multi_bias    (class_count)
        # > logits_batch       (batch_size, class_count)

        logits_batch = torch.einsum('bce, ce -> bc', mixes_batch, self.multi_weight) + self.multi_bias

        return logits_batch
