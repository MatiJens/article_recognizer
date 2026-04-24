import torch
import torch.nn as nn


class ArticleClassifierModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)
