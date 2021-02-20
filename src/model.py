import config
import transformers
from transformers import BertForNextSentencePrediction, BertForPreTraining, BertTokenizer, BertModel
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

# Let define output, out1 and out2 is the size of the vector. Ou1= last hiddent state, ou2= pooler state
    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        bo = self.bert.drop(o2)
        output = self.out(bo)
        return output
