from transformers import BertModel, BertConfig, BertTokenizer
import torch
import torch.nn as nn
from .classifier_heads import MLPHead 

class BertSentimentAnalysis(nn.Module):
    def __init__(self, num_labels: int =2, pretrained: str = None) -> None:
        super(BertSentimentAnalysis, self).__init__()
        
        if pretrained is not None:
            self.bert = BertModel.from_pretrained(pretrained)
            self.tokenizer = BertTokenizer.from_pretrained(pretrained)
            self.config = BertConfig.from_pretrained(pretrained)
        else:
            self.bert = BertModel()
            self.tokenizer = BertTokenizer()
            self.config = BertConfig()

        self.config.classifier_dropout = 0.5    
        self.classifier = MLPHead(self.config.hidden_size,
                                    self.config.hidden_size,
                                    num_labels,
                                    self.config.classifier_dropout, )

    def forward(self, inputs):
        """
            inputs: Dict, contains 'input_ids': tensor of ids of the input string after tokenizer, and 'attention_mask' for encoder
        """
        outputs = self.bert(**inputs)
        cls_tokens = outputs[1]
        logits = self.classifier(cls_tokens)
        return logits
