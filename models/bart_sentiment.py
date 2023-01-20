import torch
import torch.nn as nn
from transformers import BartConfig, BartTokenizer, BartModel
from .classifier_heads import MLPHead

class BartSentimentAnalysis(nn.Module):
    def __init__(self, num_labels=2, pretrained: str = None):
        super(BartSentimentAnalysis, self).__init__()
        if pretrained is not None:
            self.config = BartConfig.from_pretrained(pretrained)
            self.bart = BartModel.from_pretrained(pretrained)
            self.tokenizer = BartTokenizer.from_pretrained(pretrained)
            self.bart.resize_token_embeddings(len(self.tokenizer))

        else:
            self.config = BartConfig()
            self.bart = BartModel(config=self.config)
            self.tokenizer = BartTokenizer()
            self.bart.resize_token_embeddings(len(self.tokenizer))

        self.config.num_labels = num_labels

        self.classifier = MLPHead(self.config.d_model,
                                    self.config.d_model,
                                    self.config.num_labels,
                                    self.config.classifier_dropout, )

    def forward(self, tokenized_input):
        """
        tokenized_input: Dict, contains 'input_ids': tensor of ids of the input string after tokenizer, and 'attention_mask' for encoder
        """
        outputs = self.bart(**tokenized_input)
        hidden_states = outputs[0]  # last hidden states
        eos_mask = tokenized_input['input_ids'].eq(self.config.eos_token_id)
        assert len(
            torch.unique_consecutive(eos_mask.sum(1))) <= 1, "All examples must have the same number of <eos> tokens."
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                                  :, -1, :
                                  ]
        logits = self.classifier(sentence_representation)
        return logits
