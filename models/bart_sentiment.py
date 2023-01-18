import torch
import torch.nn as nn
from transformers import BartConfig, BartTokenizer, BartModel


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim: int,
            inner_dim: int,
            num_classes: int,
            pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartSentimentAnalysis(nn.Module):
    def __init__(self, config: BartConfig, pretrained: bool = False):
        super(BartSentimentAnalysis, self).__init__()
        self.config = config
        if pretrained:
            self.bart = BartModel.from_pretrained('facebook/bart-base')
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        else:
            self.bart = BartModel(config=config)
            self.tokenizer = BartTokenizer()
        self.classifier = BartClassificationHead(config.d_model,
                                                 config.d_model,
                                                 config.num_labels,
                                                 config.classifier_dropout, )

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
