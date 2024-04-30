import torch
import torch.nn as nn
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path

class LLaVa(nn.Module):
    def __init__(self, model_path):
        super(LLaVa, self).__init__()
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base="liuhaotian/llava-v1.5-7b",
            model_name=get_model_name_from_path(model_path),
            device="cuda:0"
        )
        self.tokenizer = tokenizer
        self.llava = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.model_config = model.config
        
        self.mlp = nn.ModuleList([
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        ])
        
        self.mlp = self.mlp.to("cuda:0")
        
    def forward(self, input_ids, image_tensor, attn_mask):
        outputs = self.llava(
            input_ids=input_ids,
            images=image_tensor,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        last_hidden_states = outputs.hidden_states[-1]
        outputs = last_hidden_states.mean(dim=1)
        
        for layer in self.mlp:
            outputs = layer(outputs.float().to(self.mlp[0].weight.device))
        
        return outputs