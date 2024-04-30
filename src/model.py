import torch
import torch.nn as nn
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path

class LLaVa(nn.Module):
    def __init__(self, model_path):
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base="liuhaotian/llava-v1.5-7b",
            model_name=get_model_name_from_path(model_path)
        )
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        
        self.mlp = nn.ModuleList([
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        ])
        
    def forward(self, input_ids, image_tensor, attn_mask):
        outputs = self.model(
            input_ids=input_ids,
            image_tensor=image_tensor,
            attention_mask=attn_mask
        )
        
        last_hidden_states = outputs.last_hidden_state
        outputs = last_hidden_states
        
        for layer in self.mlp:
            outputs = layer(outputs)
        
        return outputs