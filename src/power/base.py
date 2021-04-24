from typing import List, Tuple

from torch import Tensor
from torch.nn import Module, Sigmoid
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from models.ent import Ent
from models.fact import Fact
from models.pred import Pred
from models.rel import Rel


class Base(Module):
    tokenizer: DistilBertTokenizer
    bert: DistilBertForSequenceClassification

    classes: List[Tuple[Rel, Ent]]

    def __init__(self, pre_trained: str, classes: List[Tuple[Rel, Ent]]):
        super().__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained(pre_trained)
        self.tokenizer.add_tokens(['[MENTION_START]', '[MENTION_END]', '[MASK]'], special_tokens=True)

        class_count = len(classes)
        self.classes = classes

        self.bert = DistilBertForSequenceClassification.from_pretrained(pre_trained, num_labels=class_count)

    def predict(self, ent: Ent, sents: List[str]) -> List[Pred]:
        encoded = self.tokenizer(sents, padding=True, truncation=True, max_length=64, return_tensors='pt')

        self.train()

        logits_batch, softs_batch, = self.forward(encoded.input_ids.unsqueeze(0),
                                                  encoded.attention_mask.unsqueeze(0))
        logits = logits_batch[0]
        probs = Sigmoid()(logits).detach().numpy()
        softs = softs_batch[0].detach().numpy()

        pred = {Fact(ent, rel, tail): (probs[c].item(), [(sents[i], softs[c][i].item()) for i in range(len(sents))])
                for c, (rel, tail) in enumerate(self.classes) if probs[c] > 0.5}

        preds = [Pred(fact, conf, sents, []) for fact, (conf, sents) in pred.items()]

        preds.sort(key=lambda pred: pred.conf, reverse=True)

        return preds

    def forward(self, tok_lists_batch: Tensor, masks_batch: Tensor) -> Tuple[Tensor, None]:
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

        return logits_batch, None
