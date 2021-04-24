from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter, Softmax, Module, Sigmoid
from transformers import DistilBertModel, DistilBertTokenizer

from models.ent import Ent
from models.fact import Fact
from models.pred import Pred
from models.rel import Rel


class Texter(Module):
    tokenizer: DistilBertTokenizer
    bert: DistilBertModel

    class_embs: Parameter
    multi_weight: Parameter
    multi_bias: Parameter

    classes: List[Tuple[Rel, Ent]]

    use_embs: str

    def __init__(self, pre_trained: str, classes: List[Tuple[Rel, Ent]], use_embs='cls'):
        super().__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained(pre_trained)
        self.tokenizer.add_tokens(['[MENTION_START]', '[MENTION_END]', '[MASK]'], special_tokens=True)

        self.bert = DistilBertModel.from_pretrained(pre_trained)

        class_count = len(classes)
        emb_size = self.bert.config.dim

        self.class_embs = Parameter(torch.randn(class_count, emb_size))
        self.multi_weight = Parameter(torch.randn(class_count, emb_size))
        self.multi_bias = Parameter(torch.randn(class_count))

        self.classes = classes

        self.use_embs = use_embs

    def predict(self, ent: Ent, sents: List[str]) -> List[Pred]:
        encoded = self.tokenizer(sents, padding=True, truncation=True, max_length=64, return_tensors='pt')

        self.train()

        logits_batch, softs_batch, = self.forward(encoded.input_ids.unsqueeze(0), encoded.attention_mask.unsqueeze(0))
        logits = logits_batch[0]
        probs = Sigmoid()(logits).detach().numpy()
        softs = softs_batch[0].detach().numpy()

        pred = {Fact(ent, rel, tail): (probs[c].item(), [(sents[i], softs[c][i].item()) for i in range(len(sents))])
                for c, (rel, tail) in enumerate(self.classes) if probs[c] > 0.5}

        preds = [Pred(fact, conf, sents, []) for fact, (conf, sents) in pred.items()]

        preds.sort(key=lambda pred: pred.conf, reverse=True)

        return preds

    def forward(self, toks_batch: Tensor, masks_batch: Tensor) -> Tuple[Tensor, Tensor]:
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

        flat_tok_embs_batch = self.bert(input_ids=flat_tok_batch, attention_mask=flat_mask_batch).last_hidden_state

        if self.use_embs == 'cls':
            flat_sent_batch = flat_tok_embs_batch[:, 0, :]
        elif self.use_embs == 'no-cls':
            flat_mask_batch[:, 0] = 0
            flat_sent_batch = torch.mean(flat_tok_embs_batch * flat_mask_batch.unsqueeze(-1), dim=-2)
        elif self.use_embs == 'mask':
            flat_sent_batch = torch.mean(flat_tok_embs_batch * flat_mask_batch.unsqueeze(-1), dim=-2)
        elif self.use_embs == 'all':
            flat_sent_batch = flat_tok_embs_batch.mean(dim=-2)
        else:
            raise ValueError('Invalid use_embs "{}". Must be one of {}.'.format(
                self.use_embs, ['cls', 'no-cls', 'mask', 'all']))

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

        return logits_batch, softs_batch
