from transformers import LongformerModel, LongformerConfig

from longformer.sliding_chunks import pad_to_window_size
from typing import Tuple
from transformers import LongformerTokenizer
import torch
import torch.nn as nn

class LongformerMaxp(nn.Module):
    def __init__(
        self,
        pretrained: str,
        max_query_len: int,
        max_doc_len: int,
        mode: str = 'cls',
        task: str = 'ranking'
    ) -> None:
        super(LongformerMaxp, self).__init__()
        self._pretrained = pretrained
        self._max_query_len = max_query_len
        self._max_doc_len = max_doc_len
        self._mode = mode
        self._task = task
        self._config = LongformerConfig.from_pretrained(self._pretrained)
        self._config.attention_mode = 'sliding_chunks'
        self._config.gradient_checkpointing = 'True'
        #print("attention_mode: "+self._config.attention_mode)
        self._model = LongformerModel.from_pretrained(self._pretrained,config=self._config)
        self._activation = nn.ReLU()
        self.dense = nn.Linear(self._config.hidden_size, 128)
        self.dropout = nn.Dropout(self._config.hidden_dropout_prob)
        self.out_proj = nn.Linear(128, 2)

        if self._task == 'ranking':
            self._dense2 = nn.Linear(128, 1)
        elif self._task == 'classification':
            self._dense2 = nn.Linear(128, 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor = None, segment_ids: torch.Tensor = None, global_attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        num = input_ids.size()[0]
        input_ids = input_ids.view(num*3, self._max_query_len+self._max_doc_len+3)
        input_mask = input_mask.view(num*3, self._max_query_len+self._max_doc_len+3)
        global_attention_mask = global_attention_mask.view(num*3, self._max_query_len+self._max_doc_len+3)
        #input_ids, attention_mask = pad_to_window_size(input_ids, global_attention_mask, self._config.attention_window[0], self._tokenizer.pad_token_id)
        output = self._model(input_ids, attention_mask=input_mask, global_attention_mask = global_attention_mask)
        hidden_states = output[0][:, 0, :].view(num,3,-1).max(dim=1)[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states  = self.dropout(hidden_states)
        logits = self.out_proj(hidden_states)
        print("logits: "+str(logits))
        score = logits.view(-1,2)
        #logits = self._activation(self._dense1(logits.view(num,4,-1).max(dim=1)[0]))
        return score, logits